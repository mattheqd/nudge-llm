#!/usr/bin/env python3
"""
Test script to demonstrate RAG with realistic traffic simulator design scenario.
"""

import json
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from rag_inference import RAGService
from dotenv import load_dotenv

load_dotenv()

def main():
    # Load sample query
    sample_file = Path(__file__).parent / "sample_rag_query.json"
    with open(sample_file, 'r') as f:
        sample = json.load(f)
    
    print("="*70)
    print("SAMPLE RAG QUERY - Traffic Simulator Design")
    print("="*70)
    print("\n📋 QUERY:")
    print(f"   {sample['query']}")
    
    print("\n💬 CHAT HISTORY:")
    for msg in sample['chat_history']:
        role = msg['role'].upper()
        content = msg['content']
        print(f"   [{role}]: {content}")
    
    print("\n📝 SCRATCHPAD:")
    for line in sample['scratchpad'].split('\n'):
        print(f"   {line}")
    
    print("\n" + "="*70)
    print("GENERATING NUDGE WITH RAG...")
    print("="*70)
    
    # Initialize RAG service
    chunks_dir = "data/processed_chunks"
    index_dir = "data/faiss_index"
    
    rag = RAGService(chunks_dir, index_dir)
    rag.build_or_load_index()
    rag.initialize_llm()
    
    # Generate nudge
    result = rag.generate_suggestion(
        query=sample['query'],
        chat_history=sample['chat_history'],
        scratchpad=sample['scratchpad'],
        k=sample.get('k', 3)
    )
    
    print("\n" + "="*70)
    print("✨ GENERATED NUDGE PROMPT")
    print("="*70)
    print(result["suggestion"])
    
    print("\n" + "="*70)
    print("📚 REFERENCES USED")
    print("="*70)
    for i, ref in enumerate(result["references"], 1):
        print(f"\n[{i}] Source: {ref['source']}")
        print(f"    Preview: {ref['preview'][:150]}...")

if __name__ == "__main__":
    main()

