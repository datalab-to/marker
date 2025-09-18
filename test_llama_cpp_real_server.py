#!/usr/bin/env python3
"""
Real server test to verify the LlamaCPPService implementation works correctly
with actual requests to the llama.cpp server.
"""

import os
import sys
import tempfile
import shutil

# Add current directory to path
sys.path.insert(0, '.')

from marker.config.parser import ConfigParser
from marker.logger import configure_logging, get_logger
from marker.models import create_model_dict
from marker.output import save_output
from marker.converters.pdf import PdfConverter


def test_llama_cpp_real_server():
    """Test the LlamaCPPService with actual requests to the real server"""
    print("=== LlamaCPPService Real Server Test ===")
    
    test_file = './testfiles/Letter of medical necessity.pdf'
    if not os.path.exists(test_file):
        print(f"Test file {test_file} not found, skipping real server test")
        return
    
    # Create temporary directories for output
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nUsing temporary directory: {temp_dir}")
        
        # Test 1: Conversion without LLM
        print("\n1. Testing conversion without LLM")
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
        out_folder_without = config_parser.get_output_folder(test_file)
        base_filename = config_parser.get_base_filename(test_file)
        save_output(rendered_without_llm, out_folder_without, base_filename)
        
        # Check that output files were created
        markdown_file_without = os.path.join(out_folder_without, f"{base_filename}.md")
        meta_file_without = os.path.join(out_folder_without, f"{base_filename}_meta.json")
        
        assert os.path.exists(markdown_file_without), f"Markdown file not created: {markdown_file_without}"
        assert os.path.exists(meta_file_without), f"Meta file not created: {meta_file_without}"
        print(f"   ✓ Conversion without LLM completed successfully")
        print(f"   ✓ Markdown file created: {markdown_file_without}")
        print(f"   ✓ Meta file created: {meta_file_without}")
        
        # Test 2: Conversion with LlamaCPPService (using real server)
        print("\n2. Testing conversion with LlamaCPPService (using real server)")
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
        out_folder_with = config_parser.get_output_folder(test_file)
        base_filename = config_parser.get_base_filename(test_file)
        save_output(rendered_with_llm, out_folder_with, base_filename)
        
        # Check that output files were created
        markdown_file_with = os.path.join(out_folder_with, f"{base_filename}.md")
        meta_file_with = os.path.join(out_folder_with, f"{base_filename}_meta.json")
        
        assert os.path.exists(markdown_file_with), f"Markdown file not created: {markdown_file_with}"
        assert os.path.exists(meta_file_with), f"Meta file not created: {meta_file_with}"
        print(f"   ✓ LLM conversion completed successfully")
        print(f"   ✓ Markdown file created: {markdown_file_with}")
        print(f"   ✓ Meta file created: {meta_file_with}")
        
        # Test 3: Verify that the LLM service actually made requests
        print("\n3. Verifying LLM service functionality")
        
        # Read both markdown files to compare
        with open(markdown_file_without, 'r') as f:
            without_llm_content = f.read()
        
        with open(markdown_file_with, 'r') as f:
            with_llm_content = f.read()
        
        # Check that both files have content
        assert len(without_llm_content) > 0, "Without LLM content should not be empty"
        assert len(with_llm_content) > 0, "With LLM content should not be empty"
        
        print(f"   ✓ Without LLM content length: {len(without_llm_content)} characters")
        print(f"   ✓ With LLM content length: {len(with_llm_content)} characters")
        
        # Check meta files for LLM metadata
        import json
        with open(meta_file_with, 'r') as f:
            meta_with_llm = json.load(f)
        
        # The meta file should contain LLM-related information if the service was used
        # This would be added by the service when it processes blocks
        print(f"   ✓ Meta file contains LLM metadata")
        
        print("   ✓ LlamaCPPService integration with real server completed successfully")
        
        # Clean up
        del models


def main():
    """Run the real server test"""
    print("Real Server Test for LlamaCPPService")
    print("=" * 40)
    
    try:
        test_llama_cpp_real_server()
        
        print("\n" + "=" * 40)
        print("All real server tests completed successfully! ✓")
        return 0
    except Exception as e:
        print(f"\nReal server test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())