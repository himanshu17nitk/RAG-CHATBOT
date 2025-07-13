#!/usr/bin/env python3
"""
Simple Predict API Test Script
Tests the predict API endpoint with session management
"""

import requests
import json
import time
from typing import Dict, Any
from config import CLIENT_SECRET_KEY

# Configuration
BASE_URL = "http://localhost:8000"
CLIENT_SECRET = CLIENT_SECRET_KEY

def test_predict_api():
    """Test the predict API with different scenarios"""
    print("Testing PREDICT API")
    print("=" * 50)
    
    # Test 1: Basic prediction with session
    print("\nTest 1: Basic prediction with session...")
    test_basic_prediction()
    
    # Test 2: Multiple predictions in same session
    print("\nTest 2: Multiple predictions in same session...")
    test_multiple_predictions()
    
    # Test 3: Different query types
    print("\nTest 3: Different query types...")
    test_different_query_types()
    
    # Test 4: Authentication
    print("\nTest 4: Authentication...")
    test_authentication()
    
    # Test 5: Error handling
    print("\nTest 5: Error handling...")
    test_error_handling()
    
    # Test 6: Performance testing
    print("\nTest 6: Performance testing...")
    test_performance()

def create_session(user_id: str) -> str | None:
    """Create a session and return session ID"""
    url = f"{BASE_URL}/create-session"
    headers = {
        "X-Client-Secret": CLIENT_SECRET
    }
    
    data = {
        "user_id": user_id
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            session_id = result.get('session_id')
            print(f"   Session created successfully: {session_id}")
            return session_id
        else:
            print(f"   Failed to create session: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"   Error creating session: {str(e)}")
        return None

def test_basic_prediction():
    """Test basic prediction functionality"""
    # First create a session
    session_id = create_session("test_user_123")
    if not session_id:
        print("   SKIPPED - Could not create session")
        return False
    
    url = f"{BASE_URL}/predict"
    headers = {
        "X-Client-Secret": CLIENT_SECRET
    }
    
    # Test query
    test_query = "What is artificial intelligence?"
    
    data = {
        "query": test_query,
        "session_id": session_id,
        "user_id": "test_user_predict",
        "caching_flag": True
    }
    
    try:
        print(f"   Sending query: '{test_query}'")
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   SUCCESS")
            print(f"   Response: {result.get('response', 'N/A')[:100]}...")
            print(f"   Model: {result.get('model', 'N/A')}")
            print(f"   Input tokens: {result.get('input_tokens', 'N/A')}")
            print(f"   Output tokens: {result.get('output_tokens', 'N/A')}")
            print(f"   Chunks retrieved: {result.get('chunks_retrieved', 'N/A')}")
            print(f"   Chat history length: {result.get('chat_history_length', 'N/A')}")
            
            # Show timing breakdown
            timing = result.get('timing_breakdown', {})
            print(f"   Timing breakdown:")
            print(f"     - Retrieval: {timing.get('retrieval', 'N/A')}")
            print(f"     - History: {timing.get('history', 'N/A')}")
            print(f"     - Prompt: {timing.get('prompt', 'N/A')}")
            print(f"     - LLM: {timing.get('llm', 'N/A')}")
            print(f"     - Save: {timing.get('save', 'N/A')}")
            print(f"     - Total: {timing.get('total', 'N/A')}")
            
            return True
        else:
            print(f"   FAILED")
            print(f"   Status Code: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ERROR - {str(e)}")
        return False

def test_multiple_predictions():
    """Test multiple predictions in the same session"""
    # Create session
    session_id = create_session("test_user_123")
    if not session_id:
        print("   SKIPPED - Could not create session")
        return False
    
    url = f"{BASE_URL}/predict"
    headers = {
        "X-Client-Secret": CLIENT_SECRET
    }
    
    # Multiple queries to test conversation flow
    queries = [
        "What is machine learning?",
        "How does it relate to artificial intelligence?",
        "What are the main types of machine learning?",
        "Can you give me an example of supervised learning?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"   Query {i}: '{query}'")
        
        data = {
            "query": query,
            "session_id": session_id,
            "user_id": "test_user_multi",
            "caching_flag": True
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"   Response {i}: {result.get('response', 'N/A')[:80]}...")
                print(f"   Chunks: {result.get('chunks_retrieved', 'N/A')} | History: {result.get('chat_history_length', 'N/A')}")
                
                # Check if chat history is growing
                if i > 1:
                    history_length = result.get('chat_history_length', 0)
                    expected_history = (i - 1) * 2  # Each previous query adds 2 entries (query + response)
                    print(f"   History check: {history_length} (expected ~{expected_history})")
                
            else:
                print(f"   FAILED - Status: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ERROR - {str(e)}")
            return False
    
    print("   SUCCESS - All queries processed")
    return True

def test_different_query_types():
    """Test different types of queries"""
    # Create session
    session_id = create_session("test_user_123")
    if not session_id:
        print("   SKIPPED - Could not create session")
        return False
    
    url = f"{BASE_URL}/predict"
    headers = {
        "X-Client-Secret": CLIENT_SECRET
    }
    
    # Different types of queries
    query_types = [
        {
            "type": "Simple question",
            "query": "What is AI?"
        },
        {
            "type": "Complex question",
            "query": "How does deep learning differ from traditional machine learning approaches?"
        },
        {
            "type": "Technical question",
            "query": "Explain the difference between supervised and unsupervised learning with examples."
        },
        {
            "type": "Short query",
            "query": "ML"
        },
        {
            "type": "Long query",
            "query": "Can you provide a comprehensive explanation of how neural networks work, including the mathematical foundations, different types of layers, activation functions, and training processes?"
        }
    ]
    
    for query_type in query_types:
        print(f"   Testing {query_type['type']}: '{query_type['query'][:50]}...'")
        
        data = {
            "query": query_type['query'],
            "session_id": session_id,
            "user_id": "test_user_types",
            "caching_flag": True
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"   SUCCESS - Response length: {len(result.get('response', ''))} chars")
                print(f"   Tokens: {result.get('input_tokens', 'N/A')} in, {result.get('output_tokens', 'N/A')} out")
            else:
                print(f"   FAILED - Status: {response.status_code}")
                
        except Exception as e:
            print(f"   ERROR - {str(e)}")
    
    return True

def test_authentication():
    """Test authentication scenarios"""
    url = f"{BASE_URL}/predict"
    
    # Test 1: Missing client secret
    print("   Testing missing client secret...")
    data = {
        "query": "Test query",
        "session_id": "test_session",
        "user_id": "test_user",
        "caching_flag": True
    }
    
    try:
        response = requests.post(url, json=data)  # No headers
        
        if response.status_code == 401:
            print("   SUCCESS - Correctly rejected missing client secret")
        else:
            print(f"   FAILED - Should have rejected, got {response.status_code}")
            
    except Exception as e:
        print(f"   ERROR - {str(e)}")
    
    # Test 2: Invalid client secret
    print("   Testing invalid client secret...")
    headers = {
        "X-Client-Secret": "invalid_secret"
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 401:
            print("   SUCCESS - Correctly rejected invalid client secret")
        else:
            print(f"   FAILED - Should have rejected, got {response.status_code}")
            
    except Exception as e:
        print(f"   ERROR - {str(e)}")
    
    # Test 3: Valid authentication
    print("   Testing valid authentication...")
    session_id = create_session("test_user_123")
    if session_id:
        headers = {
            "X-Client-Secret": CLIENT_SECRET
        }
        data = {
            "query": "Test query",
            "session_id": session_id,
            "user_id": "test_user_auth",
            "caching_flag": True
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                print("   SUCCESS - Valid authentication accepted")
            else:
                print(f"   FAILED - Valid auth rejected: {response.status_code}")
                
        except Exception as e:
            print(f"   ERROR - {str(e)}")
    
    return True

def test_error_handling():
    """Test error handling scenarios"""
    url = f"{BASE_URL}/predict"
    headers = {
        "X-Client-Secret": CLIENT_SECRET
    }
    
    # Test 1: Empty query
    print("   Testing empty query...")
    session_id = create_session("test_user_123")
    data = {
            "query": "",
            "session_id": session_id,
            "user_id": "test_user_error",
            "caching_flag": True
        }
        
    try:
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code in [200, 400, 422]:
            print("   Handled empty query appropriately")
        else:
            print(f"   Unexpected response for empty query: {response.status_code}")
            
    except Exception as e:
        print(f"   Error with empty query: {str(e)}")
    
    # Test 2: Missing required fields
    print("   Testing missing required fields...")
    test_cases = [
        {"query": "Test query", "user_id": "test_user"},  # Missing session_id
        {"query": "Test query", "session_id": "test_session"},  # Missing user_id
        {"user_id": "test_user", "session_id": "test_session"},  # Missing query
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"     Case {i}: Missing {list(set(['query', 'session_id', 'user_id']) - set(test_case.keys()))[0]}")
        
        try:
            response = requests.post(url, headers=headers, json=test_case)
            
            if response.status_code in [400, 422]:
                print(f"     SUCCESS - Correctly rejected missing field")
            else:
                print(f"     FAILED - Should have rejected, got {response.status_code}")
                
        except Exception as e:
            print(f"     ERROR - {str(e)}")
    
    # Test 3: Invalid session ID
    print("   Testing invalid session ID...")
    data = {
        "query": "Test query",
        "session_id": "invalid_session_id",
        "user_id": "test_user_error",
        "caching_flag": True
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code in [200, 400, 404]:
            print("   Handled invalid session ID appropriately")
        else:
            print(f"   Unexpected response for invalid session: {response.status_code}")
            
    except Exception as e:
        print(f"   Error with invalid session: {str(e)}")
    
    return True

def test_performance():
    """Test performance with multiple rapid requests"""
    # Create session
    session_id = create_session("test_user_123")
    if not session_id:
        print("   SKIPPED - Could not create session")
        return False
    
    url = f"{BASE_URL}/predict"
    headers = {
        "X-Client-Secret": CLIENT_SECRET
    }
    
    # Test queries
    queries = [
        "What is AI?",
        "Explain machine learning",
        "How does deep learning work?",
        "What is natural language processing?",
        "Explain computer vision"
    ]
    
    print("   Testing performance with 5 rapid requests...")
    start_time = time.time()
    
    successful_requests = 0
    total_response_time = 0
    
    for i, query in enumerate(queries, 1):
        data = {
            "query": query,
            "session_id": session_id,
            "user_id": "test_user_perf",
            "caching_flag": True
        }
        
        try:
            request_start = time.time()
            response = requests.post(url, headers=headers, json=data)
            request_time = time.time() - request_start
            
            if response.status_code == 200:
                successful_requests += 1
                total_response_time += request_time
                result = response.json()
                
                # Get actual processing time from response
                processing_time = result.get('timing_breakdown', {}).get('total', 'N/A')
                print(f"   Request {i}: {request_time:.3f}s (processing: {processing_time})")
            else:
                print(f"   Request {i}: FAILED ({response.status_code})")
                
        except Exception as e:
            print(f"   Request {i}: ERROR - {str(e)}")
    
    total_time = time.time() - start_time
    avg_response_time = total_response_time / successful_requests if successful_requests > 0 else 0
    
    print(f"   Performance Summary:")
    print(f"     - Total time: {total_time:.3f}s")
    print(f"     - Successful requests: {successful_requests}/{len(queries)}")
    print(f"     - Average response time: {avg_response_time:.3f}s")
    print(f"     - Requests per second: {successful_requests/total_time:.2f}")
    
    return successful_requests == len(queries)

def check_server_status():
    """Check if the server is running"""
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        return True
    except:
        print("Server is not running. Please start your FastAPI server first:")
        print("uvicorn main:app --reload")
        return False

if __name__ == "__main__":
    if check_server_status():
        test_predict_api()
    else:
        print("Please start the server and try again.") 
