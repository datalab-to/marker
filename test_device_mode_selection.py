#!/usr/bin/env python3

import os
import sys
import time
from marker.config.parser import ConfigParser
from marker.models import create_model_dict
from marker.logger import configure_logging, get_logger
from marker.utils.device_mode import detect_device_mode, is_nvidia_path, is_intel_path
from marker.scripts.convert_single import convert_single_cli

configure_logging()
logger = get_logger()


def test_device_mode_selection():
    """Test the device mode selection logic"""
    logger.info("Testing device mode selection...")
    
    # Detect device mode
    device_mode = detect_device_mode()
    logger.info(f"Detected device mode: {device_mode}")
    
    # Check which path we're using
    if is_nvidia_path(device_mode):
        logger.info("Using NVIDIA path with MPS")
    elif is_intel_path(device_mode):
        logger.info("Using Intel path with continuous batching")
    else:
        logger.info(f"Using {device_mode} path")
    
    # Test model creation with device mode
    logger.info("Testing model creation with device mode...")
    try:
        models = create_model_dict(config={})
        logger.info(f"Successfully created models: {list(models.keys())}")
    except Exception as e:
        logger.error(f"Error creating models: {e}")
        return False
    
    # Test with a sample document if available
    fpath = "./testfiles/Letter of medical necessity.pdf"
    if os.path.exists(fpath):
        logger.info(f"Testing document conversion with: {fpath}")
        
        # Configuration options
        kwargs = {
            "output_dir": "./output",
            "output_format": "markdown"
        }
        
        try:
            start = time.time()
            config_parser = ConfigParser(kwargs)
            
            # Get converter class
            converter_cls = config_parser.get_converter_cls()
            
            # Create converter
            converter = converter_cls(
                config=config_parser.generate_config_dict(),
                artifact_dict=models,
                processor_list=config_parser.get_processors(),
                renderer=config_parser.get_renderer(),
                llm_service=config_parser.get_llm_service(),
            )
            
            logger.info("Starting document conversion...")
            rendered = converter(fpath)
            
            # Save output
            out_folder = config_parser.get_output_folder(fpath)
            base_name = config_parser.get_base_filename(fpath)
            os.makedirs(out_folder, exist_ok=True)
            
            # Save the rendered output
            output_path = os.path.join(out_folder, f"{base_name}.md")
            # Handle different output types
            if hasattr(rendered, 'text'):
                content = rendered.text
            else:
                content = str(rendered)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            logger.info(f"Saved markdown to {output_path}")
            logger.info(f"Total time: {time.time() - start:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        logger.warning(f"Test file not found: {fpath}")
        logger.info("Device mode selection logic tested successfully without document conversion")
    
    return True


if __name__ == "__main__":
    success = test_device_mode_selection()
    if success:
        logger.info("Device mode selection test completed successfully")
        sys.exit(0)
    else:
        logger.error("Device mode selection test failed")
        sys.exit(1)