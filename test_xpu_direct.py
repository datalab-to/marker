#!/usr/bin/env python3
"""
Direct API test for XPU document processing
"""

import os
import sys
import time

# Add current directory to path
sys.path.insert(0, '.')

from marker.config.parser import ConfigParser
from marker.logger import configure_logging, get_logger
from marker.models import create_model_dict
from marker.output import save_output

# Configure logging
configure_logging()
logger = get_logger()
logger.setLevel('DEBUG')

def test_direct_api():
    print("Starting direct API conversion test...")
    
    # Create model dict
    print("Creating model dict...")
    models = create_model_dict()
    print("Model dict created:", list(models.keys()))
    
    # Create config parser with command line arguments
    kwargs = {
        'output_dir': './output',
        'output_format': 'markdown'
    }
    print("Creating config parser with kwargs:", kwargs)
    config_parser = ConfigParser(kwargs)
    print("Config parser created")
    
    # Get converter class
    converter_cls = config_parser.get_converter_cls()
    print("Converter class:", converter_cls)
    
    # Generate config dict
    config_dict = config_parser.generate_config_dict()
    print("Config dict:", config_dict)
    
    # Get other parameters
    processors = config_parser.get_processors()
    print("Processors:", processors)
    renderer = config_parser.get_renderer()
    print("Renderer:", renderer)
    llm_service = config_parser.get_llm_service()
    print("LLM service:", llm_service)
    
    # Create converter
    print("Creating converter...")
    converter = converter_cls(
        config=config_dict,
        artifact_dict=models,
        processor_list=processors,
        renderer=renderer,
        llm_service=llm_service,
    )
    print("Converter created")
    
    # Convert file
    fpath = './testfiles/Letter of medical necessity.pdf'
    print("Converting file:", fpath)
    start = time.time()
    try:
        rendered = converter(fpath)
        print("Conversion completed in", time.time() - start, "seconds")
        
        # Save output
        out_folder = config_parser.get_output_folder(fpath)
        print("Output folder:", out_folder)
        base_filename = config_parser.get_base_filename(fpath)
        print("Base filename:", base_filename)
        save_output(rendered, out_folder, base_filename)
        print("Output saved")
        
        logger.info(f"Saved markdown to {out_folder}")
        logger.info(f"Total time: {time.time() - start}")
        
        print("Test completed successfully!")
        return True
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_direct_api()
    sys.exit(0 if success else 1)