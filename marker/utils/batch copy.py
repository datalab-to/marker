import asyncio
import logging
from typing import Any, Callable, Optional

import torch

from marker.utils.gpu import GPUManager
from marker.utils.device_mode import detect_device_mode, is_nvidia_path, is_intel_path


def get_batch_sizes_worker_counts(gpu_manager: GPUManager, peak_worker_vram: int):
    # Detect device mode
    device_mode = detect_device_mode()
    
    # For Intel XPU path, use continuous batching with single model
    if is_intel_path(device_mode):
        # Use smaller batch sizes and single worker for continuous batching
        return {
            "layout_batch_size": 6,
            "detection_batch_size": 4,
            "table_rec_batch_size": 6,
            "ocr_error_batch_size": 6,
            "recognition_batch_size": 32,
            "equation_batch_size": 8,
            "detector_postprocessing_cpu_workers": 1,
        }, 1  # Single worker for continuous batching
    
    # For NVIDIA path, use existing MPS behavior
    vram = gpu_manager.get_gpu_vram()
    workers = max(1, vram // peak_worker_vram)
    
    if workers == 1:
        return {}, workers
    
    return {
        "layout_batch_size": 12,
        "detection_batch_size": 8,
        "table_rec_batch_size": 12,
        "ocr_error_batch_size": 12,
        "recognition_batch_size": 64,
        "equation_batch_size": 16,
        "detector_postprocessing_cpu_workers": 2,
    }, workers


def derive_batch_plan_for_intel(gpu_manager: GPUManager, peak_worker_vram: int):
    """
    Derive batch plan for Intel/XPU path ONLY.
    
    This function uses the existing get_batch_sizes_worker_counts function's output
    but derives batch_size values for Intel from the same outputs.
    
    Args:
        gpu_manager (GPUManager): GPU manager instance
        peak_worker_vram (int): Peak VRAM per worker in GB
        
    Returns:
        dict: Batch plan with per_model_batch_size, legacy_workers, and aux
    """
    model_overrides, workers = get_batch_sizes_worker_counts(gpu_manager, peak_worker_vram)
    # Per your request: treat the old "workers" as the initial target batch size.
    # If overrides contain explicit batch sizes for certain models, prefer them;
    # otherwise default to `workers`. No caps, no warm-up auto-tuning.
    per_model_batch_size = {
        # example mapping using existing override keys when present
        "layout_model": model_overrides.get("layout_batch_size", workers),
        "detection_model": model_overrides.get("detection_batch_size", workers),
        "table_rec_model": model_overrides.get("table_rec_batch_size", workers),
        "ocr_error_model": model_overrides.get("ocr_error_batch_size", workers),
        "recognition_model": model_overrides.get("recognition_batch_size", workers),
        "equation_model": model_overrides.get("equation_batch_size", workers),
    }
    # pass through detector_postprocessing_cpu_workers if needed by your pipeline
    aux = {"detector_postprocessing_cpu_workers": model_overrides.get("detector_postprocessing_cpu_workers", 0)}
    # Return both sets so callers can use either path without changing the legacy function
    return {
        "per_model_batch_size": per_model_batch_size,
        "legacy_workers": workers,
        "aux": aux,
    }


class Batcher:
    """
    A class for continuous batching of model inference requests.
    
    This class manages a queue of inference requests and batches them together
    for efficient processing. It uses a timer-based flush mechanism to ensure
    responsiveness while maximizing batch size.
    """
    
    def __init__(
        self,
        model: Any,
        device: torch.device,
        max_batch_size: int,
        flush_timeout_ms: int,
        collate_fn: Callable,
        unbatch_fn: Callable,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Batcher.
        
        Args:
            model: The model to use for inference
            device: The device to run inference on
            max_batch_size: Maximum number of items in a batch
            flush_timeout_ms: Timeout in milliseconds before flushing the queue
            collate_fn: Function to collate items into a batch
            unbatch_fn: Function to split batch output back into individual items
            logger: Optional logger for debugging
        """
        self.model = model
        self.device = device
        self.max_batch_size = max_batch_size
        self.flush_timeout_ms = flush_timeout_ms / 1000.0  # Convert to seconds
        self.collate_fn = collate_fn
        self.unbatch_fn = unbatch_fn
        self.logger = logger or logging.getLogger(__name__)
        
        # Internal state
        self.queue = []  # List of (item, future) tuples
        self.timer_handle = None
        self.running = False
    
    async def submit(self, item) -> asyncio.Future:
        """
        Submit an item for batched processing.
        
        Args:
            item: The item to process
            
        Returns:
            A future that will be resolved with the result
        """
        future = asyncio.Future()
        self.queue.append((item, future))
        
        # Start the timer if this is the first item
        if len(self.queue) == 1:
            self._start_timer()
        
        return future
    
    def _start_timer(self):
        """Start the flush timer."""
        if self.timer_handle:
            self.timer_handle.cancel()
        self.timer_handle = asyncio.get_event_loop().call_later(
            self.flush_timeout_ms, self._timer_fired
        )
    
    def _timer_fired(self):
        """Called when the timer fires."""
        # Process a batch if we have items
        if self.queue:
            asyncio.create_task(self._process_batch())
    
    async def _process_batch(self):
        """Process a batch of items."""
        if not self.queue:
            return
            
        # Gather items for the batch (up to max_batch_size)
        batch_items = self.queue[:self.max_batch_size]
        self.queue = self.queue[self.max_batch_size:]
        
        # Extract items and futures
        items = [item for item, _ in batch_items]
        futures = [future for _, future in batch_items]
        
        try:
            # Collate items into a batch
            batch = self.collate_fn(items)
            
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch = [b.to(self.device) if hasattr(b, 'to') else b for b in batch]
            elif hasattr(batch, 'to'):
                batch = batch.to(self.device)
            
            # Run inference
            with torch.inference_mode():
                outputs = self.model(batch)
            
            # Split outputs and resolve futures
            results = self.unbatch_fn(outputs)
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)
                    
        except Exception as e:
            # Set exception on all futures
            for future in futures:
                if not future.done():
                    future.set_exception(e)
            self.logger.error(f"Batch processing failed: {e}", exc_info=True)
        
        # Restart timer if queue is not empty
        if self.queue:
            self._start_timer()
    
    async def run_forever(self):
        """
        Run the batcher indefinitely, processing batches as they become ready.
        
        This method should be called once to start the batcher's processing loop.
        """
        self.running = True
        self.logger.info("Batcher started")
        
        try:
            while self.running:
                # Wait for items to be added to the queue
                while not self.queue and self.running:
                    await asyncio.sleep(0.001)  # Small sleep to prevent busy waiting
                
                if not self.running:
                    break
                
                # Start timer when first item arrives
                self._start_timer()
                
                # Wait until either max_batch_size items are queued or timer fires
                while (len(self.queue) < self.max_batch_size and
                       self.timer_handle and not self.timer_handle.cancelled() and
                       self.running):
                    await asyncio.sleep(0.001)
                
                # Cancel timer as we're about to process
                if self.timer_handle:
                    self.timer_handle.cancel()
                    self.timer_handle = None
                
                # Process a batch if we have items
                if self.queue:
                    await self._process_batch()
        except Exception as e:
            self.logger.error(f"Batcher error: {e}", exc_info=True)
            raise
        finally:
            self.running = False
            self.logger.info("Batcher stopped")
    
    
    def stop(self):
        """Stop the batcher."""
        self.running = False
        if self.timer_handle:
            self.timer_handle.cancel()
            self.timer_handle = None
    
    def submit_sync(self, item):
        """
        Submit an item for batched processing synchronously.
        
        Args:
            item: The item to process
            
        Returns:
            The result of the processing
        """
        import asyncio
        import threading
        
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop in current thread, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Submit the item and get a future
        future = asyncio.run_coroutine_threadsafe(self.submit(item), loop)
        
        # Wait for the result
        return future.result()
