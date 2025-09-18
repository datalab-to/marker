#!/usr/bin/env python3

import os
import sys
import time
from marker.config.parser import ConfigParser
from marker.models import create_model_dict
from marker.logger import configure_logging, get_logger

configure_logging()
logger = get_logger()

def test_xpu_document():
    # Path to your test document
    fpath = "./testfiles/Letter of medical necessity.pdf"
    
    # Check if file exists
    if not os.path.exists(fpath):
        logger.error(f"File not found: {fpath}")
        return
    
    logger.info(f"Testing XPU document processing with: {fpath}")
    
    # Configuration options for LlamaCPPService
    kwargs = {
        "llm_service": "marker.services.llama_cpp.LlamaCPPService",
        "use_llm": True,
        "llama_cpp_model": "NuMarkdown",
        "llama_cpp_base_url": "http://192.168.68.186:8080",
        "output_dir": ".",
        "output_format": "markdown"
    }
    
    try:
        # Create models
        models = create_model_dict(config={})
        
        start = time.time()
        config_parser = ConfigParser(kwargs)
        
        # Get converter class
        converter_cls = config_parser.get_converter_cls()
        
        # Create converter with LlamaCPPService
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
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(rendered)
        
        logger.info(f"Saved markdown to {output_path}")
        logger.info(f"Total time: {time.time() - start:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_xpu_document()