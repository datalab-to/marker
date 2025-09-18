#!/usr/bin/env python3
"""
Example usage of the LlamaCPPService with various configurations.
This demonstrates how to use the service in different scenarios.
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, '.')

from marker.services.llama_cpp import LlamaCPPService
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from pydantic import BaseModel


class ExampleSchema(BaseModel):
    """Example schema for LLM response"""
    summary: str
    key_points: list[str]


def demonstrate_direct_usage():
    """Demonstrate direct usage of the LlamaCPPService"""
    print("=== Direct Usage Examples ===")
    
    # Example 1: Basic usage with default configuration
    print("\n1. Basic usage with default configuration")
    service = LlamaCPPService()
    print(f"   Base URL: {service.llama_cpp_base_url}")
    print(f"   Model: {service.llama_cpp_model}")
    print("   ✓ Service created with default configuration")
    
    # Example 2: Custom configuration
    print("\n2. Custom configuration")
    service_custom = LlamaCPPService({
        'llama_cpp_base_url': 'http://192.168.68.186:8080',
        'llama_cpp_model': 'NuMarkdown',
        'timeout': 60,
        'max_retries': 3
    })
    print(f"   Base URL: {service_custom.llama_cpp_base_url}")
    print(f"   Model: {service_custom.llama_cpp_model}")
    print(f"   Timeout: {service_custom.timeout}")
    print(f"   Max Retries: {service_custom.max_retries}")
    print("   ✓ Service created with custom configuration")
    
    # Example 3: Usage with environment variables
    print("\n3. Usage with environment variables")
    # In a real scenario, these would be set before importing the service
    os.environ['LLAMA_CPP_BASE_URL'] = 'http://192.168.68.186:8080'
    os.environ['LLAMA_CPP_MODEL'] = 'NuMarkdown'
    
    # Note: Environment variables are read at import time, so we'd need to reload
    # the module to see the effect. For demonstration, we'll just show the concept.
    print("   Environment variables would be used at import time")
    print("   Base URL from env: http://192.168.68.186:8080")
    print("   Model from env: NuMarkdown")
    
    # Clean up
    if 'LLAMA_CPP_BASE_URL' in os.environ:
        del os.environ['LLAMA_CPP_BASE_URL']
    if 'LLAMA_CPP_MODEL' in os.environ:
        del os.environ['LLAMA_CPP_MODEL']


def demonstrate_integration_with_converter():
    """Demonstrate integration with PdfConverter"""
    print("\n=== Integration with PdfConverter ===")
    
    # Create model dict
    model_dict = create_model_dict()
    
    # Example 1: Converter with LlamaCPPService
    print("\n1. Converter with LlamaCPPService")
    converter = PdfConverter(
        artifact_dict=model_dict,
        config={
            "use_llm": True,
            "llm_service": "marker.services.llama_cpp.LlamaCPPService",
            "llama_cpp_base_url": "http://192.168.68.186:8080",
            "llama_cpp_model": "NuMarkdown",
        },
        llm_service="marker.services.llama_cpp.LlamaCPPService"
    )
    
    print(f"   LLM Service Type: {type(converter.llm_service).__name__}")
    print(f"   LLM Service String: {converter.llm_service}")
    print("   ✓ Converter successfully integrated with LlamaCPPService")
    
    # Clean up
    del model_dict


def demonstrate_cli_usage():
    """Demonstrate CLI usage patterns"""
    print("\n=== CLI Usage Patterns ===")
    
    print("\n1. CLI command without LLM:")
    print("   marker_single /path/to/document.pdf")
    
    print("\n2. CLI command with LLM (auto-enabled):")
    print("   marker_single /path/to/document.pdf --llm_service marker.services.llama_cpp.LlamaCPPService")
    
    print("\n3. CLI command with LLM and explicit enable:")
    print("   marker_single /path/to/document.pdf --use_llm --llm_service marker.services.llama_cpp.LlamaCPPService")
    
    print("\n4. CLI command with custom configuration:")
    print("   marker_single /path/to/document.pdf \\")
    print("     --llm_service marker.services.llama_cpp.LlamaCPPService \\")
    print("     --llama_cpp_base_url http://192.168.68.186:8080 \\")
    print("     --llama_cpp_model NuMarkdown")
    
    print("\n5. Environment variables for configuration:")
    print("   export LLAMA_CPP_BASE_URL=http://192.168.68.186:8080")
    print("   export LLAMA_CPP_MODEL=NuMarkdown")
    print("   marker_single /path/to/document.pdf --llm_service marker.services.llama_cpp.LlamaCPPService")


def main():
    """Run all examples"""
    print("LlamaCPPService Usage Examples")
    print("=" * 40)
    
    try:
        demonstrate_direct_usage()
        demonstrate_integration_with_converter()
        demonstrate_cli_usage()
        
        print("\n" + "=" * 40)
        print("All usage examples completed successfully! ✓")
        return 0
    except Exception as e:
        print(f"\nExample failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())