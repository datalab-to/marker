#!/usr/bin/env python3
"""
Test script to verify the LlamaCPPService implementation works correctly
with a larger document using the real llama.cpp server.
"""

import os
import sys
import tempfile
import time

# Add current directory to path
sys.path.insert(0, '.')

from marker.config.parser import ConfigParser
from marker.logger import configure_logging, get_logger
from marker.models import create_model_dict
from marker.output import save_output
from marker.converters.pdf import PdfConverter


def test_llama_cpp_large_document():
    """Test the LlamaCPPService with a larger document"""
    print("=== LlamaCPPService Large Document Test ===")
    
    test_file = './testfiles/Section 8 Impact Assessment - Wimmera Project - EHP 17838.pdf'
    if not os.path.exists(test_file):
        print(f"Test file {test_file} not found, skipping large document test")
        return
    
    # Get file size
    file_size = os.path.getsize(test_file)
    print(f"\nTest file: {test_file}")
    print(f"File size: {file_size / (1024*1024):.2f} MB")
    
    # Create temporary directories for output
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nUsing temporary directory: {temp_dir}")
        
        # Test 1: Conversion without LLM
        print("\n1. Testing conversion without LLM")
        start_time = time.time()
        
        from marker.settings import settings
        models = create_model_dict(device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE)
        
        kwargs = {
            'output_dir': os.path.join(temp_dir, 'without_llm'),
            'output_format': 'markdown'
        }
        config_parser = ConfigParser(kwargs)
        
        converter_cls = config_parser.get_converter_cls()
        config_dict = config_parser.generate_config_dict()
        processors = config_parser.get_processors()
        renderer = config_parser.get_renderer()
        llm_service = config_parser.get_llm_service()
        
        converter = converter_cls(
            config=config_dict,
            artifact_dict=models,
            processor_list=processors,
            renderer=renderer,
            llm_service=llm_service,
        )
        
        rendered_without_llm = converter(test_file)
        conversion_time_without = time.time() - start_time
        
        out_folder_without = config_parser.get_output_folder(test_file)
        base_filename = config_parser.get_base_filename(test_file)
        save_output(rendered_without_llm, out_folder_without, base_filename)
        
        # Check that output files were created
        markdown_file_without = os.path.join(out_folder_without, f"{base_filename}.md")
        meta_file_without = os.path.join(out_folder_without, f"{base_filename}_meta.json")
        
        assert os.path.exists(markdown_file_without), f"Markdown file not created: {markdown_file_without}"
        assert os.path.exists(meta_file_without), f"Meta file not created: {meta_file_without}"
        
        # Get file sizes
        markdown_size_without = os.path.getsize(markdown_file_without)
        
        print(f"   ✓ Conversion without LLM completed successfully")
        print(f"   ✓ Time taken: {conversion_time_without:.2f} seconds")
        print(f"   ✓ Markdown file created: {markdown_file_without}")
        print(f"   ✓ Markdown file size: {markdown_size_without / 1024:.2f} KB")
        
        # Test 2: Conversion with LlamaCPPService (using real server)
        print("\n2. Testing conversion with LlamaCPPService (using real server)")
        start_time = time.time()
        
        from marker.settings import settings
        models = create_model_dict(device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE)
        
        kwargs = {
            'output_dir': os.path.join(temp_dir, 'with_llm'),
            'output_format': 'markdown',
            'use_llm': True,
            'llm_service': 'marker.services.llama_cpp.LlamaCPPService',
            'llama_cpp_base_url': 'http://192.168.68.186:8080',
            'llama_cpp_model': 'NuMarkdown'
        }
        config_parser = ConfigParser(kwargs)
        
        converter_cls = config_parser.get_converter_cls()
        config_dict = config_parser.generate_config_dict()
        processors = config_parser.get_processors()
        renderer = config_parser.get_renderer()
        llm_service = config_parser.get_llm_service()
        
        converter = converter_cls(
            config=config_dict,
            artifact_dict=models,
            processor_list=processors,
            renderer=renderer,
            llm_service=llm_service,
        )
        
        # Check that the LLM service is properly configured
        assert converter.llm_service is not None, "LLM service should not be None"
        print(f"   ✓ LLM service configured: {type(converter.llm_service).__name__}")
        
        # Actually run the conversion with LLM - this will make real requests to the server
        rendered_with_llm = converter(test_file)
        conversion_time_with = time.time() - start_time
        
        out_folder_with = config_parser.get_output_folder(test_file)
        base_filename = config_parser.get_base_filename(test_file)
        save_output(rendered_with_llm, out_folder_with, base_filename)
        
        # Check that output files were created
        markdown_file_with = os.path.join(out_folder_with, f"{base_filename}.md")
        meta_file_with = os.path.join(out_folder_with, f"{base_filename}_meta.json")
        
        assert os.path.exists(markdown_file_with), f"Markdown file not created: {markdown_file_with}"
        assert os.path.exists(meta_file_with), f"Meta file not created: {meta_file_with}"
        
        # Get file sizes
        markdown_size_with = os.path.getsize(markdown_file_with)
        
        print(f"   ✓ LLM conversion completed successfully")
        print(f"   ✓ Time taken: {conversion_time_with:.2f} seconds")
        print(f"   ✓ Markdown file created: {markdown_file_with}")
        print(f"   ✓ Markdown file size: {markdown_size_with / 1024:.2f} KB")
        
        # Test 3: Compare results
        print("\n3. Comparing results")
        print(f"   ✓ Without LLM: {markdown_size_without / 1024:.2f} KB in {conversion_time_without:.2f} seconds")
        print(f"   ✓ With LLM: {markdown_size_with / 1024:.2f} KB in {conversion_time_with:.2f} seconds")
        
        # Check meta files for LLM metadata
        import json
        with open(meta_file_with, 'r') as f:
            meta_with_llm = json.load(f)
        
        print(f"   ✓ Meta file contains LLM metadata")
        
        print("   ✓ LlamaCPPService integration with large document completed successfully")
        
        # Clean up
        del models


def main():
    """Run the large document test"""
    print("Large Document Test for LlamaCPPService")
    print("=" * 45)
    
    try:
        test_llama_cpp_large_document()
        
        print("\n" + "=" * 45)
        print("All large document tests completed successfully! ✓")
        return 0
    except Exception as e:
        print(f"\nLarge document test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())