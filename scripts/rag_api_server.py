#!/usr/bin/env python3
"""
Flask API server for RAG queries - can be called from Node.js backend.
Usage: python scripts/rag_api_server.py --port 5000
"""

import os
import sys
import json
import argparse
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from rag_inference import RAGService

load_dotenv()

app = Flask(__name__)
CORS(app)  # Allow CORS for Node.js backend

# Initialize RAG service (singleton)
rag_service = None

def get_rag_service():
    """Get or initialize RAG service."""
    global rag_service
    if rag_service is None:
        chunks_dir = os.getenv("CHUNKS_DIR", "data/processed_chunks")
        index_dir = os.getenv("FAISS_INDEX_DIR", "data/faiss_index")
        rag_service = RAGService(chunks_dir, index_dir)
        rag_service.build_or_load_index()
        rag_service.initialize_llm()
    return rag_service

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})

@app.route("/api/rag/suggest", methods=["POST"])
def rag_suggest():
    """Generate gentle nudge prompt using RAG."""
    try:
        data = request.json
        query = data.get("query")
        chat_history = data.get("chat_history")
        scratchpad = data.get("scratchpad")
        k = data.get("k", 3)
        
        if not query:
            return jsonify({"error": "query is required"}), 400
        
        rag = get_rag_service()
        result = rag.generate_suggestion(
            query=query,
            chat_history=chat_history,
            scratchpad=scratchpad,
            k=k
        )
        
        return jsonify({
            "suggestion": result["suggestion"],
            "nudge": result.get("nudge", result["suggestion"]),  # Support both field names
            "references": result["references"]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/rag/rebuild-index", methods=["POST"])
def rebuild_index():
    """Rebuild FAISS index from chunks."""
    try:
        global rag_service
        chunks_dir = os.getenv("CHUNKS_DIR", "data/processed_chunks")
        index_dir = os.getenv("FAISS_INDEX_DIR", "data/faiss_index")
        rag_service = RAGService(chunks_dir, index_dir)
        rag_service.build_or_load_index(force_rebuild=True)
        rag_service.initialize_llm()
        
        return jsonify({"status": "Index rebuilt successfully"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG API Server")
    parser.add_argument("--port", type=int, default=5000, help="Port to run server on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    
    args = parser.parse_args()
    
    print(f"Starting RAG API server on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)

