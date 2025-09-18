import os
import asyncio
import threading
from typing import Dict, Any

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = (
    "1"  # Transformers uses .isin for an op, which is not supported on MPS
)

from surya.foundation import FoundationPredictor
from surya.detection import DetectionPredictor
from surya.layout import LayoutPredictor
from surya.ocr_error import OCRErrorPredictor
from surya.recognition import RecognitionPredictor
from surya.table_rec import TableRecPredictor

from marker.logger import get_logger
from marker.utils.device_mode import detect_device_mode, is_intel_path
from marker.settings import settings
from marker.utils.gpu import GPUManager
from marker.utils.batch import Batcher, derive_batch_plan_for_intel

logger = get_logger()


def _create_batcher_for_model(model, device, max_batch_size, model_name, flush_timeout_ms=10):
    """Create a Batcher instance for a model with appropriate collate and unbatch functions."""
    
    def default_collate_fn(items):
        """Default collate function - assumes items are already in the right format."""
        return items[0] if len(items) == 1 else items
    
    def default_unbatch_fn(outputs):
        """Default unbatch function - assumes outputs are a list of results."""
        if not isinstance(outputs, (list, tuple)):
            return [outputs]
        return outputs
    
    # Create batcher with default functions - these might need to be customized per model type
    batcher = Batcher(
        model=model,
        device=device,
        max_batch_size=max_batch_size,
        flush_timeout_ms=flush_timeout_ms,
        collate_fn=default_collate_fn,
        unbatch_fn=default_unbatch_fn,
        logger=logger
    )
    
    return batcher


def _start_batcher_thread(batcher):
    """Start a batcher's run_forever method in a separate thread with its own event loop."""
    
    def run_batcher_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(batcher.run_forever())
        except Exception as e:
            logger.error(f"Batcher thread error: {e}", exc_info=True)
        finally:
            loop.close()
    
    thread = threading.Thread(target=run_batcher_in_thread, daemon=True)
    thread.start()
    return thread


def cleanup_model_dict(models: dict):
    """Clean up batcher threads and resources when models are no longer needed."""
    if "_batchers" in models:
        batchers = models["_batchers"]
        for batcher in batchers.values():
            try:
                batcher.stop()
            except Exception as e:
                logger.warning(f"Error stopping batcher: {e}")
    
    # Note: Thread cleanup is handled by daemon threads which will terminate when main program exits
    # If explicit thread cleanup is needed, it would need to be implemented here


def create_model_dict(
    device=None, dtype=None, attention_implementation: str | None = None, config: dict | None = None
) -> dict:
    # If device is not specified, detect the appropriate device mode
    if device is None:
        device_mode = detect_device_mode()
        device = device_mode
        
        # For Intel path, we might want to use specific optimizations
        if is_intel_path(device_mode):
            logger.info("Using Intel XPU path with continuous batching optimization")
        else:
            logger.info(f"Using {device_mode} path")
    
    # Get configuration values with defaults
    flush_timeout_ms = config.get("flush_timeout_ms", 5) if config else 5
    intel_batching = config.get("intel_batching", None) if config else None
    
    # If intel_batching is not explicitly set, default to True when XPU is active
    if intel_batching is None:
        intel_batching = is_intel_path(device) if device else False
    
    foundation_predictor = FoundationPredictor(
        device=device, dtype=dtype, attention_implementation=attention_implementation
    )
    
    # Create base models
    layout_model = LayoutPredictor(device=device, dtype=dtype)
    recognition_model = RecognitionPredictor(foundation_predictor)
    table_rec_model = TableRecPredictor(device=device, dtype=dtype)
    detection_model = DetectionPredictor(device=device, dtype=dtype)
    ocr_error_model = OCRErrorPredictor(device=device, dtype=dtype)
    
    models = {
        "foundation_model": foundation_predictor,
        "layout_model": layout_model,
        "recognition_model": recognition_model,
        "table_rec_model": table_rec_model,
        "detection_model": detection_model,
        "ocr_error_model": ocr_error_model,
    }
    
    # For Intel path, create batchers for each model if intel_batching is enabled
    if device is not None and is_intel_path(device) and intel_batching:
        logger.info("Creating batchers for Intel XPU path")
        
        # Create GPU manager to get batch plan
        gpu_manager = GPUManager(0)  # Use device 0 for XPU
        batch_plan = derive_batch_plan_for_intel(gpu_manager, gpu_manager.default_gpu_vram)
        per_model_batch_sizes = batch_plan["per_model_batch_size"]
        
        # Create batchers for each model
        batchers = {}
        batcher_threads = {}
        
        # Layout model
        layout_batcher = _create_batcher_for_model(
            layout_model,
            device,
            per_model_batch_sizes.get("layout_model", 6),
            "layout_model",
            flush_timeout_ms
        )
        batchers["layout_model"] = layout_batcher
        batcher_threads["layout_model"] = _start_batcher_thread(layout_batcher)
        
        # Detection model
        detection_batcher = _create_batcher_for_model(
            detection_model,
            device,
            per_model_batch_sizes.get("detection_model", 4),
            "detection_model",
            flush_timeout_ms
        )
        batchers["detection_model"] = detection_batcher
        batcher_threads["detection_model"] = _start_batcher_thread(detection_batcher)
        
        # Table recognition model
        table_rec_batcher = _create_batcher_for_model(
            table_rec_model,
            device,
            per_model_batch_sizes.get("table_rec_model", 6),
            "table_rec_model",
            flush_timeout_ms
        )
        batchers["table_rec_model"] = table_rec_batcher
        batcher_threads["table_rec_model"] = _start_batcher_thread(table_rec_batcher)
        
        # OCR error model
        ocr_error_batcher = _create_batcher_for_model(
            ocr_error_model,
            device,
            per_model_batch_sizes.get("ocr_error_model", 6),
            "ocr_error_model",
            flush_timeout_ms
        )
        batchers["ocr_error_model"] = ocr_error_batcher
        batcher_threads["ocr_error_model"] = _start_batcher_thread(ocr_error_batcher)
        
        # Recognition model (special case as it's based on foundation predictor)
        recognition_batcher = _create_batcher_for_model(
            recognition_model,
            device,
            per_model_batch_sizes.get("recognition_model", 32),
            "recognition_model",
            flush_timeout_ms
        )
        batchers["recognition_model"] = recognition_batcher
        batcher_threads["recognition_model"] = _start_batcher_thread(recognition_batcher)
        
        # Store batchers and threads in the models dict
        models["_batchers"] = batchers
        models["_batcher_threads"] = batcher_threads
        
        logger.info("Batchers created and started for Intel XPU path")
    
    return models
