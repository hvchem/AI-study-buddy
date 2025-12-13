#!/usr/bin/env python3
"""
Test script to verify AI Study Buddy basic functionality.
This script tests the core components without requiring the full API to be running.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.pdf_processor import PDFProcessor
from app.core.config import settings

def test_text_chunking():
    """Test the text chunking functionality."""
    print("=" * 60)
    print("Testing Text Chunking")
    print("=" * 60)
    
    processor = PDFProcessor()
    
    # Sample text
    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    as opposed to natural intelligence displayed by animals including humans. 
    Leading AI textbooks define the field as the study of intelligent agents: 
    any system that perceives its environment and takes actions that maximize 
    its chance of achieving its goals. Some popular accounts use the term 
    artificial intelligence to describe machines that mimic cognitive functions 
    that humans associate with the human mind, such as learning and problem solving.
    """ * 10  # Repeat to get enough text for chunking
    
    chunks = processor.chunk_text(sample_text)
    
    print(f"\nOriginal text length: {len(sample_text.split())} words")
    print(f"Number of chunks created: {len(chunks)}")
    print(f"Chunk size setting: {settings.chunk_size} words")
    print(f"Chunk overlap setting: {settings.chunk_overlap} words")
    
    if chunks:
        print(f"\nFirst chunk preview (first 100 chars):")
        print(f"  {chunks[0][:100]}...")
        
        if len(chunks) > 1:
            print(f"\nSecond chunk preview (first 100 chars):")
            print(f"  {chunks[1][:100]}...")
    
    print("\n✅ Text chunking test passed!")
    return True


def test_pdf_processor_initialization():
    """Test that PDF processor initializes correctly."""
    print("\n" + "=" * 60)
    print("Testing PDF Processor Initialization")
    print("=" * 60)
    
    processor = PDFProcessor()
    
    print(f"\nChunk size: {processor.chunk_size}")
    print(f"Chunk overlap: {processor.chunk_overlap}")
    print(f"Upload directory: {settings.upload_dir}")
    
    # Check if upload directory exists
    if os.path.exists(settings.upload_dir):
        print(f"✅ Upload directory exists: {settings.upload_dir}")
    else:
        print(f"⚠️  Upload directory will be created on first use: {settings.upload_dir}")
    
    print("\n✅ PDF processor initialization test passed!")
    return True


def test_configuration():
    """Test that configuration is loaded correctly."""
    print("\n" + "=" * 60)
    print("Testing Configuration")
    print("=" * 60)
    
    print(f"\nApp Name: {settings.app_name}")
    print(f"Version: {settings.version}")
    print(f"API Host: {settings.api_host}")
    print(f"API Port: {settings.api_port}")
    print(f"Embedding Model: {settings.embedding_model}")
    print(f"LLM Model: {settings.llm_model}")
    print(f"Max Upload Size: {settings.max_upload_size / 1024 / 1024} MB")
    print(f"FAISS Index Path: {settings.faiss_index_path}")
    
    print("\n✅ Configuration test passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("AI STUDY BUDDY - Component Tests")
    print("=" * 60)
    
    tests = [
        ("Configuration", test_configuration),
        ("PDF Processor Initialization", test_pdf_processor_initialization),
        ("Text Chunking", test_text_chunking),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with error: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
