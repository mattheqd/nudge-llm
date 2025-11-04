#!/usr/bin/env python3
"""
Test script for the RAG pipeline.
Tests document processing, indexing, and API endpoints.
"""

import os
import sys
import json
import requests
import time
from pathlib import Path
from dotenv import load_dotenv

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))
from rag_inference import RAGService

load_dotenv()

def test_document_processing():
    """Test Step 1: Document extraction and chunking."""
    print("\n" + "="*60)
    print("STEP 1: Testing Document Processing")
    print("="*60)
    
    # Check if we have sample PDFs or need to create test data
    textbooks_dir = Path("data/textbooks")
    chunks_dir = Path("data/processed_chunks")
    
    if not textbooks_dir.exists():
        print(f"⚠️  {textbooks_dir} directory doesn't exist.")
        print("   Create it and add PDF files, or create sample text chunks.")
        print("\n   For quick testing, creating a sample chunk file...")
        
        # Create sample chunks for testing
        chunks_dir.mkdir(parents=True, exist_ok=True)
        sample_chunk = {
            "chunk_id": 0,
            "text": "To implement caching in a web application, use Redis or Memcached for in-memory storage. Cache frequently accessed data like user profiles and session data. Set appropriate TTL (time-to-live) values to ensure data freshness. Consider cache invalidation strategies when data is updated.",
            "token_count": 45,
            "source_file": "sample_design_patterns.pdf"
        }
        
        sample_file = chunks_dir / "sample_chunks.jsonl"
        with open(sample_file, 'w') as f:
            json.dump(sample_chunk, f)
            f.write('\n')
        
        print(f"✅ Created sample chunk file: {sample_file}")
        return True
    else:
        pdf_files = list(textbooks_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"⚠️  No PDF files found in {textbooks_dir}")
            print("   Add PDF files there or use sample chunks for testing.")
            return False
        
        print(f"✅ Found {len(pdf_files)} PDF file(s)")
        print(f"   Run: python scripts/extract_text.py --input {textbooks_dir} --output {chunks_dir}/")
        return True

def test_rag_service():
    """Test Step 2: RAG service initialization and indexing."""
    print("\n" + "="*60)
    print("STEP 2: Testing RAG Service")
    print("="*60)
    
    try:
        chunks_dir = os.getenv("CHUNKS_DIR", "data/processed_chunks")
        index_dir = os.getenv("FAISS_INDEX_DIR", "data/faiss_index")
        
        print(f"Loading RAG service...")
        print(f"  Chunks dir: {chunks_dir}")
        print(f"  Index dir: {index_dir}")
        
        rag = RAGService(chunks_dir, index_dir)
        
        print("\nBuilding/loading FAISS index...")
        success = rag.build_or_load_index()
        if not success:
            print("❌ Failed to build index. Make sure you have processed documents first.")
            return False
        
        print("✅ Index built/loaded successfully")
        
        print("\nInitializing ZotGPT LLM...")
        rag.initialize_llm()
        print("✅ LLM initialized")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_rag_query():
    """Test Step 3: Query the RAG service directly."""
    print("\n" + "="*60)
    print("STEP 3: Testing RAG Query (Direct)")
    print("="*60)
    
    try:
        chunks_dir = os.getenv("CHUNKS_DIR", "data/processed_chunks")
        index_dir = os.getenv("FAISS_INDEX_DIR", "data/faiss_index")
        
        rag = RAGService(chunks_dir, index_dir)
        rag.build_or_load_index()
        rag.initialize_llm()
        
        query = "How should I implement caching?"
        chat_history = [{"role": "user", "content": "My API is getting slow"}]
        scratchpad = "Need low latency, handle 10k concurrent users"
        
        print(f"Query: {query}")
        print(f"Chat history: {len(chat_history)} messages")
        print(f"Scratchpad: {scratchpad}")
        print("\nGenerating suggestion...")
        
        result = rag.generate_suggestion(
            query=query,
            chat_history=chat_history,
            scratchpad=scratchpad,
            k=3
        )
        
        print("\n✅ Suggestion generated:")
        print("-" * 60)
        print(result["suggestion"])
        print("-" * 60)
        print(f"\n📚 References: {len(result['references'])} chunks retrieved")
        for i, ref in enumerate(result["references"], 1):
            print(f"  [{i}] Source: {ref['source']}")
            print(f"      Preview: {ref['preview'][:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_server():
    """Test Step 4: Test API server endpoints."""
    print("\n" + "="*60)
    print("STEP 4: Testing API Server")
    print("="*60)
    
    api_url = os.getenv("RAG_API_URL", "http://localhost:5000")
    
    print(f"Testing API at: {api_url}")
    print("\n⚠️  Make sure the API server is running:")
    print("   python scripts/rag_api_server.py --port 5000")
    
    # Test health endpoint
    print(f"\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ Health check passed: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {api_url}")
        print("   Make sure the API server is running!")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test suggest endpoint
    print(f"\n2. Testing /api/rag/suggest endpoint...")
    try:
        payload = {
            "query": "How should I implement caching?",
            "chat_history": [{"role": "user", "content": "My API is slow"}],
            "scratchpad": "Need low latency",
            "k": 3
        }
        
        response = requests.post(
            f"{api_url}/api/rag/suggest",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Suggestion received:")
            print("-" * 60)
            print(result["suggestion"])
            print("-" * 60)
            print(f"\n📚 References: {len(result.get('references', []))} chunks")
            return True
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("RAG Pipeline Test Suite")
    print("="*60)
    
    results = {
        "Document Processing": test_document_processing(),
        "RAG Service": test_rag_service(),
        "RAG Query": test_rag_query(),
    }
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "="*60)
    print("API SERVER TEST (Manual)")
    print("="*60)
    print("Start the API server in a separate terminal:")
    print("  python scripts/rag_api_server.py --port 5000")
    print("\nThen run this script again or test manually with:")
    print("  curl -X POST http://localhost:5000/api/rag/suggest \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"query\": \"How to implement caching?\"}'")
    
    if all(results.values()):
        print("\n✅ All core tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()

