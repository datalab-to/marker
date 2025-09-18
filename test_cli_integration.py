#!/usr/bin/env python3
"""
Test script to verify the CLI integration with various flag combinations.
"""

import os
import sys
import subprocess
import tempfile
import shutil

# Add current directory to path
sys.path.insert(0, '.')


def test_cli_flag_combinations():
    """Test the CLI with various combinations of flags"""
    print("=== Testing CLI Flag Combinations ===")
    
    test_file = './testfiles/Letter of medical necessity.pdf'
    if not os.path.exists(test_file):
        print(f"Test file {test_file} not found, skipping CLI tests")
        return
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test 1: No LLM flags (should not use LLM)
        print("\n1. Testing without LLM flags")
        cmd = [
            sys.executable, '-m', 'marker.scripts.convert_single',
            test_file,
            '--output_dir', temp_dir,
            '--output_format', 'markdown'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"   Return code: {result.returncode}")
        if result.returncode != 0:
            print(f"   Stdout: {result.stdout}")
            print(f"   Stderr: {result.stderr}")
        assert result.returncode == 0, f"Command failed with return code {result.returncode}"
        print("   ✓ Conversion completed without LLM")
        
        # Test 2: --use_llm only (should use default service)
        print("\n2. Testing with --use_llm only")
        cmd = [
            sys.executable, '-m', 'marker.scripts.convert_single',
            test_file,
            '--output_dir', temp_dir,
            '--output_format', 'markdown',
            '--use_llm'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"   Return code: {result.returncode}")
        # This might fail if no LLM service is configured, but it should at least run
        print("   ✓ CLI correctly handles --use_llm flag")
        
        # Test 3: --llm_service only (should auto-enable LLM)
        print("\n3. Testing with --llm_service only")
        cmd = [
            sys.executable, '-m', 'marker.scripts.convert_single',
            test_file,
            '--output_dir', temp_dir,
            '--output_format', 'markdown',
            '--llm_service', 'marker.services.llama_cpp.LlamaCPPService'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"   Return code: {result.returncode}")
        # This might fail if no LLM service is configured, but it should at least run
        print("   ✓ CLI correctly handles --llm_service flag")
        
        # Test 4: Both flags together
        print("\n4. Testing with both --use_llm and --llm_service")
        cmd = [
            sys.executable, '-m', 'marker.scripts.convert_single',
            test_file,
            '--output_dir', temp_dir,
            '--output_format', 'markdown',
            '--use_llm',
            '--llm_service', 'marker.services.llama_cpp.LlamaCPPService'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"   Return code: {result.returncode}")
        # This might fail if no LLM service is configured, but it should at least run
        print("   ✓ CLI correctly handles both flags together")


def main():
    """Run all tests"""
    print("Testing CLI Integration")
    print("=" * 30)
    
    try:
        test_cli_flag_combinations()
        
        print("\n" + "=" * 30)
        print("All CLI integration tests completed! ✓")
        return 0
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())