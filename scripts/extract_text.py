#!/usr/bin/env python3
"""
Extract text from PDF textbooks and chunk into ~512 token pieces.
Usage: python scripts/extract_text.py --input data/textbooks/book.pdf --output data/processed_chunks/
"""

import argparse
import json
import os
from pathlib import Path
from pypdf import PdfReader
from transformers import AutoTokenizer
from tqdm import tqdm

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, tokenizer, max_tokens=512, overlap=50):
    """Chunk text into ~max_tokens pieces with overlap."""
    # Tokenize the entire text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    chunks = []
    chunk_id = 0
    i = 0
    
    while i < len(tokens):
        # Take up to max_tokens tokens
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        
        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "token_count": len(chunk_tokens),
            "start_token": i,
            "end_token": i + len(chunk_tokens)
        })
        
        chunk_id += 1
        # Move forward with overlap
        i += max_tokens - overlap
    
    return chunks

def save_chunks(chunks, output_dir, source_file):
    """Save chunks as JSONL file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use source filename for output
    source_name = Path(source_file).stem
    output_file = output_dir / f"{source_name}_chunks.jsonl"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            chunk["source_file"] = source_file
            json.dump(chunk, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Saved {len(chunks)} chunks to {output_file}")
    return str(output_file)

def main():
    parser = argparse.ArgumentParser(description="Extract and chunk PDF text")
    parser.add_argument("--input", required=True, help="Input PDF file or directory")
    parser.add_argument("--output", required=True, help="Output directory for chunks")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per chunk")
    parser.add_argument("--overlap", type=int, default=50, help="Token overlap between chunks")
    parser.add_argument("--tokenizer", default="gpt2", help="Tokenizer model name")
    
    args = parser.parse_args()
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get PDF files
    input_path = Path(args.input)
    if input_path.is_file():
        pdf_files = [input_path]
    elif input_path.is_dir():
        pdf_files = list(input_path.glob("*.pdf"))
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return
    
    if not pdf_files:
        print(f"No PDF files found in {args.input}")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s)")
    
    # Process each PDF
    all_chunks = []
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        print(f"\nProcessing: {pdf_file.name}")
        text = extract_text_from_pdf(pdf_file)
        chunks = chunk_text(text, tokenizer, args.max_tokens, args.overlap)
        save_chunks(chunks, args.output, str(pdf_file))
        all_chunks.extend(chunks)
    
    print(f"\nTotal chunks created: {len(all_chunks)}")

if __name__ == "__main__":
    main()

