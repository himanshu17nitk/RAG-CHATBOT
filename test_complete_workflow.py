#!/usr/bin/env python3
"""
Interactive Complete Workflow Test Script
Trains HealthBot.pdf from data directory and provides interactive querying
"""

import requests
import json
import time
import os
from typing import Dict, Any
from config import CLIENT_SECRET_KEY

# Configuration
BASE_URL = "http://localhost:8000"
CLIENT_SECRET = CLIENT_SECRET_KEY
USER_ID = "himanshu"
DATA_FILE = "data/HealthBot.pdf"

def test_complete_workflow():
    """Test the complete RAG workflow with interactive querying"""
    print("Interactive Complete RAG Workflow Test")
    print("=" * 50)
    
    # Step 1: Train the HealthBot.pdf file
    print(f"\nStep 1: Training {DATA_FILE}...")
    if not train_healthbot_file():
        print("   ❌ FAILED - File training failed")
        return False
    
    print("   ✅ SUCCESS - File trained successfully!")
    
    # Step 2: Interactive session management
    print("\nStep 2: Interactive Session Management")
    session_id = handle_session_management()
    if not session_id:
        print("   ❌ FAILED - Session creation failed")
        return False
    
    # Step 3: Interactive querying
    print("\nStep 3: Interactive Querying")
    print("   Type 'quit' to exit, 'help' for commands")
    interactive_querying(session_id)
    
    return True

def train_healthbot_file():
    """Train the HealthBot.pdf file from data directory"""
    if not os.path.exists(DATA_FILE):
        print(f"   ❌ ERROR - File not found: {DATA_FILE}")
        return False
    
    url = f"{BASE_URL}/train"
    headers = {
        "X-Client-Secret": CLIENT_SECRET
    }
    
    try:
        print(f"   Uploading {DATA_FILE}...")
        
        with open(DATA_FILE, 'rb') as file:
            files = {
                'file': (os.path.basename(DATA_FILE), file, 'application/pdf')
            }
            data = {
                'user_id': USER_ID
            }
            
            response = requests.post(url, headers=headers, files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            chunks = result.get('processing_summary', {}).get('total_chunks', 0)
            file_size = result.get('processing_summary', {}).get('file_size', 0)
            
            print(f"   ✅ Training completed successfully!")
            print(f"   📊 Summary:")
            print(f"     - File: {os.path.basename(DATA_FILE)}")
            print(f"     - Size: {file_size:,} bytes")
            print(f"     - Chunks created: {chunks}")
            
            # Show timing breakdown
            timing = result.get('processing_summary', {}).get('timing_breakdown', {})
            print(f"   ⏱️  Timing breakdown:")
            print(f"     - File read: {timing.get('file_read', 'N/A')}")
            print(f"     - MongoDB save: {timing.get('mongo_save', 'N/A')}")
            print(f"     - Chunking: {timing.get('chunking', 'N/A')}")
            print(f"     - Metadata: {timing.get('metadata', 'N/A')}")
            print(f"     - Embedding & Vector storage: {timing.get('vector_storage', 'N/A')}")
            print(f"     - Total: {timing.get('total', 'N/A')}")
            
            return True
        else:
            print(f"   ❌ FAILED - Status: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR - {str(e)}")
        return False

def handle_session_management():
    """Handle session creation or continuation"""
    while True:
        print("\n   Session Options:")
        print("   1. Create new session")
        print("   2. Continue with existing session (if available)")
        print("   3. Exit")
        
        choice = input("\n   Enter your choice (1-3): ").strip()
        
        if choice == "1":
            session_id = create_session()
            if session_id:
                print(f"   ✅ New session created: {session_id}")
                return session_id
            else:
                print("   ❌ Failed to create session")
                continue
                
        elif choice == "2":
            session_id = input("   Enter existing session ID: ").strip()
            if session_id:
                # Test if session is valid
                if test_session_validity(session_id):
                    print(f"   ✅ Using existing session: {session_id}")
                    return session_id
                else:
                    print("   ❌ Invalid session ID")
                    continue
            else:
                print("   ❌ No session ID provided")
                continue
                
        elif choice == "3":
            print("   👋 Exiting...")
            return None
            
        else:
            print("   ❌ Invalid choice. Please enter 1, 2, or 3.")

def create_session():
    """Create a new session"""
    url = f"{BASE_URL}/create-session"
    headers = {
        "X-Client-Secret": CLIENT_SECRET
    }
    
    data = {
        "user_id": USER_ID
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            return result.get('session_id')
        else:
            print(f"   ❌ Failed to create session: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"   ❌ Error creating session: {str(e)}")
        return None

def test_session_validity(session_id):
    """Test if a session ID is valid by making a simple query"""
    url = f"{BASE_URL}/predict"
    headers = {
        "X-Client-Secret": CLIENT_SECRET
    }
    
    data = {
        "query": "test",
        "session_id": session_id,
        "user_id": USER_ID,
        "caching_flag": True
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        return response.status_code == 200
    except:
        return False

def interactive_querying(session_id):
    """Interactive querying interface"""
    print(f"\n   🚀 Ready for queries! Session: {session_id}")
    print("   Commands:")
    print("     - Type your question and press Enter")
    print("     - 'help' - Show this help")
    print("     - 'stats' - Show session statistics")
    print("     - 'quit' - Exit the program")
    print("     - 'clear' - Clear the screen")
    
    query_count = 0
    
    while True:
        try:
            # Get user input
            query = input(f"\n   [{query_count + 1}] Query: ").strip()
            
            if not query:
                continue
                
            # Handle commands
            if query.lower() == 'quit':
                print("   👋 Goodbye!")
                break
            elif query.lower() == 'help':
                print("   Commands:")
                print("     - Type your question and press Enter")
                print("     - 'help' - Show this help")
                print("     - 'stats' - Show session statistics")
                print("     - 'quit' - Exit the program")
                print("     - 'clear' - Clear the screen")
                continue
            elif query.lower() == 'stats':
                show_session_stats(session_id)
                continue
            elif query.lower() == 'clear':
                os.system('clear' if os.name == 'posix' else 'cls')
                print("   🚀 Ready for queries! Session: {session_id}")
                continue
            
            # Process the query
            print("   🔍 Processing query...")
            result = process_query(session_id, query)
            
            if result:
                query_count += 1
                display_query_result(result, query_count)
            else:
                print("   ❌ Failed to process query")
                
        except KeyboardInterrupt:
            print("\n   👋 Goodbye!")
            break
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

def process_query(session_id, query):
    """Process a single query"""
    url = f"{BASE_URL}/predict"
    headers = {
        "X-Client-Secret": CLIENT_SECRET
    }
    
    data = {
        "query": query,
        "session_id": session_id,
        "user_id": USER_ID,
        "caching_flag": True
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"   ❌ API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"   ❌ Network Error: {str(e)}")
        return None

def display_query_result(result, query_count):
    """Display query results in a formatted way"""
    print(f"\n   📝 Response #{query_count}:")
    print("   " + "="*60)
    
    # Display the main response
    response_text = result.get('response', 'No response received')
    print(f"   {response_text}")
    
    print("\n   📊 Details:")
    print(f"     - Model: {result.get('model', 'N/A')}")
    print(f"     - Input tokens: {result.get('input_tokens', 'N/A')}")
    print(f"     - Output tokens: {result.get('output_tokens', 'N/A')}")
    print(f"     - Chunks retrieved: {result.get('chunks_retrieved', 'N/A')}")
    print(f"     - Chat history length: {result.get('chat_history_length', 'N/A')}")
    
    # Display timing breakdown
    timing = result.get('timing_breakdown', {})
    print(f"   ⏱️  Timing:")
    print(f"     - Retrieval: {timing.get('retrieval', 'N/A')}")
    print(f"     - History: {timing.get('history', 'N/A')}")
    print(f"     - Prompt: {timing.get('prompt', 'N/A')}")
    print(f"     - LLM: {timing.get('llm', 'N/A')}")
    print(f"     - Save: {timing.get('save', 'N/A')}")
    print(f"     - Total: {timing.get('total', 'N/A')}")
    
    print("   " + "="*60)

def show_session_stats(session_id):
    """Show session statistics"""
    print(f"\n   📈 Session Statistics:")
    print(f"     - Session ID: {session_id}")
    print(f"     - User ID: {USER_ID}")
    print(f"     - Trained file: {os.path.basename(DATA_FILE)}")
    print(f"     - File size: {os.path.getsize(DATA_FILE):,} bytes")
    
    # Test a simple query to get current stats
    result = process_query(session_id, "test")
    if result:
        print(f"     - Current chat history length: {result.get('chat_history_length', 'N/A')}")
        print(f"     - Model being used: {result.get('model', 'N/A')}")

def check_server_status():
    """Check if the server is running"""
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        return True
    except:
        print("❌ Server is not running. Please start your FastAPI server first:")
        print("   uvicorn main:app --reload")
        return False

def check_data_file():
    """Check if the data file exists"""
    if not os.path.exists(DATA_FILE):
        print(f"❌ Data file not found: {DATA_FILE}")
        print("   Please ensure the HealthBot.pdf file is in the data/ directory")
        return False
    return True

if __name__ == "__main__":
    print("🏥 HealthBot RAG System - Interactive Test")
    print("=" * 50)
    
    # Check prerequisites
    if not check_server_status():
        exit(1)
    
    if not check_data_file():
        exit(1)
    
    # Run the complete workflow
    try:
        success = test_complete_workflow()
        if success:
            print("\n✅ Complete workflow test completed successfully!")
        else:
            print("\n❌ Complete workflow test failed!")
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}") 