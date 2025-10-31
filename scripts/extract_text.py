#!/usr/bin/env python3
"""
Extract text from PDF textbooks and chunk into ~512 token pieces.
Usage: python scripts/extract_text.py --input data/textbooks/book.pdf --output data/processed_chunks/
"""

# TODO: Implement PDF extraction and chunking
# - Use pypdf to extract text from PDFs
# - Use transformers tokenizer to chunk into ~512 token pieces
# - Save chunks as JSONL with metadata (chunk_id, text, tokens, topic)

if __name__ == "__main__":
    print("Placeholder: extract_text.py - PDF extraction and chunking")

