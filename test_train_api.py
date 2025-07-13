#!/usr/bin/env python3
"""
Simple Train API Test Script
Tests only the train API endpoint
"""

import requests
import json
import time
from typing import Dict, Any
from config import CLIENT_SECRET_KEY

# Configuration
BASE_URL = "http://localhost:8000"
CLIENT_SECRET = CLIENT_SECRET_KEY

def test_train_api():
    """Test the train API with different scenarios"""
    print("Testing TRAIN API")
    print("=" * 50)
    
    # Test 1: Simple document
    print("\nTest 1: Simple AI document...")
    test_simple_document()
    
    # Test 2: Larger document
    print("\nTest 2: Larger document...")
    test_larger_document()
    
    # Test 3: Different file types
    print("\nTest 3: Different file types...")
    test_different_file_types()
    
    # Test 4: Authentication
    print("\nTest 4: Authentication...")
    test_authentication()
    
    # Test 5: Error handling
    print("\nTest 5: Error handling...")
    test_error_handling()

def test_simple_document():
    """Test with a simple AI document"""
    url = f"{BASE_URL}/train"
    headers = {
        "X-Client-Secret": CLIENT_SECRET
    }
    
    # Create a simple test document about AI
    test_content = """
    Artificial Intelligence (AI) is a branch of computer science.
    It aims to create machines that can perform tasks requiring human intelligence.
    Machine learning is a subset of AI that enables computers to learn from data.
    Deep learning uses neural networks to process complex patterns.
    Natural language processing helps computers understand human language.
    """
    
    files = {
        'file': ('ai_document.txt', test_content, 'text/plain')
    }
    data = {
        'user_id': 'test_user_123'
    }
    
    try:
        response = requests.post(url, headers=headers, files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"SUCCESS")
            print(f"   Status: {result.get('status')}")
            print(f"   Message: {result.get('message')}")
            print(f"   Chunks: {result.get('processing_summary', {}).get('total_chunks', 'N/A')}")
            print(f"   File size: {result.get('processing_summary', {}).get('file_size', 'N/A')} bytes")
            return True
        else:
            print(f"FAILED")
            print(f"   Status Code: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"ERROR - {str(e)}")
        return False

def test_larger_document():
    """Test with a larger document"""
    url = f"{BASE_URL}/train"
    headers = {
        "X-Client-Secret": CLIENT_SECRET
    }
    
    # Create a larger test document
    large_content = """
    This is a larger test document for the RAG system.
    It contains multiple paragraphs to test how the system handles larger files.
    The document discusses various topics including technology, science, and business.
    """ * 50  # Repeat 50 times to make it larger
    
    files = {
        'file': ('large_document.txt', large_content, 'text/plain')
    }
    data = {
        'user_id': 'test_user_123'
    }
    
    try:
        response = requests.post(url, headers=headers, files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"SUCCESS")
            print(f"   Status: {result.get('status')}")
            print(f"   Chunks: {result.get('processing_summary', {}).get('total_chunks', 'N/A')}")
            print(f"   File size: {result.get('processing_summary', {}).get('file_size', 'N/A')} bytes")
            return True
        elif response.status_code == 413:
            print(f"EXPECTED - Document too large (413)")
            return True
        else:
            print(f"FAILED")
            print(f"   Status Code: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"ERROR - {str(e)}")
        return False

def test_different_file_types():
    """Test with different file types"""
    url = f"{BASE_URL}/train"
    headers = {
        "X-Client-Secret": CLIENT_SECRET
    }
    
    # Test with different content types
    test_cases = [
        {
            'name': 'Python code',
            'content': '''
def hello_world():
    """Simple hello world function"""
    print("Hello, World!")
    return "Hello, World!"

class Calculator:
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b
            ''',
            'filename': 'python_code.py',
            'content_type': 'text/plain'
        },
        {
            'name': 'JSON data',
            'content': '''
{
    "name": "Test Document",
    "description": "A test document for RAG system",
    "topics": ["AI", "Machine Learning", "Data Science"],
    "author": "Test User",
    "version": "1.0"
}
            ''',
            'filename': 'data.json',
            'content_type': 'application/json'
        },
        {
            'name': 'Markdown document',
            'content': '''
# Test Document

## Introduction
This is a test document in Markdown format.

## Features
- Feature 1
- Feature 2
- Feature 3

## Conclusion
This document tests the RAG system's ability to handle Markdown.
            ''',
            'filename': 'document.md',
            'content_type': 'text/markdown'
        }
    ]
    
    for test_case in test_cases:
        print(f"   Testing {test_case['name']}...")
        
        files = {
            'file': (test_case['filename'], test_case['content'], test_case['content_type'])
        }
        data = {
            'user_id': 'test_user_123'
        }
        
        try:
            response = requests.post(url, headers=headers, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"   {test_case['name']}: SUCCESS")
                print(f"      Chunks: {result.get('processing_summary', {}).get('total_chunks', 'N/A')}")
            else:
                print(f"   {test_case['name']}: FAILED ({response.status_code})")
                
        except Exception as e:
            print(f"   {test_case['name']}: ERROR - {str(e)}")

def test_authentication():
    """Test authentication scenarios"""
    url = f"{BASE_URL}/train"
    
    # Test 1: No authentication header
    print("   Testing without authentication header...")
    files = {
        'file': ('test.txt', 'Test content', 'text/plain')
    }
    data = {
        'user_id': 'test_user_123'
    }
    
    response = requests.post(url, files=files, data=data)
    
    if response.status_code == 401:
        print("   Correctly rejected without authentication")
    else:
        print(f"   Should have returned 401, got {response.status_code}")
    
    # Test 2: Invalid client secret
    print("   Testing with invalid client secret...")
    headers = {
        "X-Client-Secret": "invalid-secret-key"
    }
    
    response = requests.post(url, headers=headers, files=files, data=data)
    
    if response.status_code == 401:
        print("   Correctly rejected invalid client secret")
    else:
        print(f"   Should have returned 401, got {response.status_code}")

def test_error_handling():
    """Test error handling scenarios"""
    url = f"{BASE_URL}/train"
    headers = {
        "X-Client-Secret": CLIENT_SECRET
    }
    
    # Test 1: Empty file
    print("   Testing with empty file...")
    files = {
        'file': ('empty.txt', '', 'text/plain')
    }
    data = {
        'user_id': 'test_user_123'
    }
    
    try:
        response = requests.post(url, headers=headers, files=files, data=data)
        
        if response.status_code in [200, 400, 422]:
            print("   Handled empty file appropriately")
        else:
            print(f"   Unexpected response for empty file: {response.status_code}")
            
    except Exception as e:
        print(f"   Error with empty file: {str(e)}")
    
    # Test 2: Missing user_id
    print("   Testing without user_id...")
    files = {
        'file': ('test.txt', 'Test content', 'text/plain')
    }
    data = {}  # No user_id
    
    try:
        response = requests.post(url, headers=headers, files=files, data=data)
        
        if response.status_code in [400, 422]:
            print("   Correctly rejected missing user_id")
        else:
            print(f"   Should have rejected missing user_id, got {response.status_code}")
            
    except Exception as e:
        print(f"   Error with missing user_id: {str(e)}")

def check_server():
    """Check if server is running"""
    try:
        response = requests.get(f"{BASE_URL}/docs")
        if response.status_code == 200:
            print("Server is running!")
            return True
        else:
            print("Server not responding properly")
            return False
    except:
        print("Server is not running. Please start your FastAPI server first:")
        print("uvicorn main:app --reload")
        return False

def main():
    """Main function"""
    print("TRAIN API TEST SCRIPT")
    print("=" * 50)
    
    # Check if server is running
    if not check_server():
        return
    
    # Run all tests
    test_train_api()
    
    print("\n" + "=" * 50)
    print("Train API Testing Complete!")
    print("Check the results above to see if your train API is working correctly.")

if __name__ == "__main__":
    main() 
    