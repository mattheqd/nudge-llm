#!/usr/bin/env python3
"""
LangChain + FAISS retrieval with ZotGPT (Azure OpenAI).
Usage: python scripts/rag_inference.py --query "user question" --chat_history "..." --scratchpad "..."
"""

import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

load_dotenv()

class RAGService:
    def __init__(self, chunks_dir="data/processed_chunks", index_dir="data/faiss_index"):
        self.chunks_dir = Path(chunks_dir)
        self.index_dir = Path(index_dir)
        self.vectorstore = None
        self.llm = None
        
    def load_chunks(self):
        """Load chunks from JSONL files."""
        chunks = []
        if not self.chunks_dir.exists():
            print(f"Warning: {self.chunks_dir} does not exist. No chunks loaded.")
            return chunks
        
        for jsonl_file in self.chunks_dir.glob("*.jsonl"):
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        chunk = json.loads(line)
                        chunks.append(Document(
                            page_content=chunk["text"],
                            metadata={
                                "chunk_id": chunk.get("chunk_id", 0),
                                "source_file": chunk.get("source_file", ""),
                                "token_count": chunk.get("token_count", 0)
                            }
                        ))
        
        print(f"Loaded {len(chunks)} chunks")
        return chunks
    
    def build_or_load_index(self, force_rebuild=False):
        """Build FAISS index from chunks or load existing."""
        # Check for existing index files (FAISS saves both index.faiss and index.pkl)
        index_faiss = self.index_dir / "index.faiss"
        index_pkl = self.index_dir / "index.pkl"
        
        # FAISS requires both files to load properly
        has_index = index_faiss.exists() and index_pkl.exists()
        
        if has_index and not force_rebuild:
            print(f"Loading existing FAISS index from {self.index_dir}")
            print(f"  Found: {index_faiss.name} ({index_faiss.stat().st_size} bytes)")
            print(f"  Found: {index_pkl.name} ({index_pkl.stat().st_size} bytes)")
            
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            try:
                # FAISS.load_local needs the directory path, not the file path
                self.vectorstore = FAISS.load_local(
                    str(self.index_dir), 
                    embeddings,
                    allow_dangerous_deserialization=True  # Required for loading pickled data
                )
                print("✅ Index loaded successfully")
                return True
            except Exception as e:
                print(f"❌ Error loading index: {e}")
                print("   This might be due to incompatible index format or missing files.")
                print("   Rebuilding index...")
                # Fall through to rebuild
        
        # Build new index if needed
        print("Building new FAISS index...")
        chunks = self.load_chunks()
        
        if not chunks:
            print("No chunks found. Please run extract_text.py first.")
            return False
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = FAISS.from_documents(chunks, embeddings)
        
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(self.index_dir))
        print(f"✅ Index saved to {self.index_dir}")
        
        return True
    
    def initialize_llm(self):
        """Initialize Azure OpenAI LLM (ZotGPT)."""
        api_key = os.getenv("API_KEY")
        deployment_id = os.getenv("DEPLOYMENT_ID")
        api_version = os.getenv("API_VERSION", "2023-05-15")
        azure_endpoint = os.getenv("AZURE_ENDPOINT", "https://azureapi.zotgpt.uci.edu")
        
        if not api_key or not deployment_id:
            raise ValueError("API_KEY and DEPLOYMENT_ID must be set in environment variables")
        
        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            deployment_name=deployment_id,
            openai_api_version=api_version,
            openai_api_key=api_key,
            temperature=1.0,
            max_tokens=150  # Reduced for short nudge prompts (1-2 sentences)
        )
        print("ZotGPT LLM initialized")
    
    def format_chat_history(self, chat_history):
        """Format chat history string into messages."""
        if not chat_history:
            return []
        
        try:
            # If it's a JSON string, parse it
            if isinstance(chat_history, str):
                messages = json.loads(chat_history)
            else:
                messages = chat_history
            
            # Convert to format expected by prompt
            formatted = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    formatted.append(f"User: {content}")
                elif role == "assistant":
                    formatted.append(f"Assistant: {content}")
            
            return "\n".join(formatted)
        except:
            # If parsing fails, treat as plain text
            return chat_history
    
    def generate_suggestion(self, query, chat_history=None, scratchpad=None, k=3):
        """Generate design suggestion using RAG."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call build_or_load_index() first.")
        
        if not self.llm:
            self.initialize_llm()
        
        # Retrieve relevant chunks
        # Use similarity_search directly for compatibility with all LangChain versions
        relevant_docs = self.vectorstore.similarity_search(query, k=k)
        
        # Build context from retrieved chunks
        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            context_parts.append(f"[Reference {i}]\n{doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        # Format chat history
        chat_context = self.format_chat_history(chat_history) if chat_history else ""
        
        # Build prompt for generating gentle nudge prompts
        prompt_template = """You are a software design mentor providing gentle, thought-provoking nudges to guide designers in their thinking process.

These nudges should be:
- Short, gentle questions or prompts (1-2 sentences max)
- Thought-provoking rather than prescriptive
- Encouraging reflection on the problem, solution, or design process


IMPORTANT: Generate ONLY a short nudge question or prompt (1-2 sentences). Do not provide full design suggestions or detailed explanations. Just the gentle nudge itself.

{context_section}

{chat_section}

{scratchpad_section}

Based on the above context, chat history, and scratchpad notes, generate a single gentle nudge prompt (a reflective question or gentle suggestion) that would help the designer think more deeply about the following query:

Query: {query}

Nudge Prompt (just the question/prompt, 1-2 sentences max):"""
        
        context_section = f"Relevant Knowledge:\n{context}" if context else ""
        chat_section = f"Chat History:\n{chat_context}" if chat_context else ""
        scratchpad_section = f"Scratchpad Notes:\n{scratchpad}" if scratchpad else ""
        
        prompt = prompt_template.format(
            context_section=context_section,
            chat_section=chat_section,
            scratchpad_section=scratchpad_section,
            query=query
        )
        
        # Generate response using HumanMessage format
        response = self.llm.invoke([HumanMessage(content=prompt)]).content
        
        # Return response with references
        references = [
            {
                "chunk_id": doc.metadata.get("chunk_id", i),
                "source": doc.metadata.get("source_file", "unknown"),
                "preview": doc.page_content[:200] + "..."
            }
            for i, doc in enumerate(relevant_docs, 1)
        ]
        
        return {
            "suggestion": response,
            "nudge": response,  # Alias for consistency
            "references": references
        }

def main():
    parser = argparse.ArgumentParser(description="Generate gentle nudge prompts using RAG with ZotGPT")
    parser.add_argument("--query", required=True, help="User query or design context")
    parser.add_argument("--chat-history", default=None, help="Chat history (JSON string)")
    parser.add_argument("--scratchpad", default=None, help="Scratchpad notes")
    parser.add_argument("--chunks-dir", default="data/processed_chunks", help="Chunks directory")
    parser.add_argument("--index-dir", default="data/faiss_index", help="FAISS index directory")
    parser.add_argument("--k", type=int, default=3, help="Number of chunks to retrieve")
    
    args = parser.parse_args()
    
    # Initialize RAG service
    rag = RAGService(args.chunks_dir, args.index_dir)
    rag.build_or_load_index()
    rag.initialize_llm()
    
    # Generate suggestion
    result = rag.generate_suggestion(
        args.query,
        args.chat_history,
        args.scratchpad,
        args.k
    )
    
    print("\n" + "="*60)
    print("NUDGE PROMPT")
    print("="*60)
    print(result["suggestion"])
    print("\n" + "="*60)
    print("REFERENCES")
    print("="*60)
    for ref in result["references"]:
        print(f"\n[Reference {ref['chunk_id']}]")
        print(f"Source: {ref['source']}")
        print(f"Preview: {ref['preview']}")

if __name__ == "__main__":
    main()

