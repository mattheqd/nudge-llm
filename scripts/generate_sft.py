#!/usr/bin/env python3
"""
Generate synthetic SFT pairs from textbook chunks using OpenAI.
Usage: python scripts/generate_sft.py --chunks data/processed_chunks/ --output data/synthetic_sft/ --num_pairs 1000
"""

# TODO: Implement synthetic SFT generation
# - Load chunks from data/processed_chunks/
# - For each chunk, use OpenAI API to generate synthetic context + suggestion
# - Format: {context, scratchpad, instruction, output}
# - Generate ~1000 pairs, save to data/synthetic_sft/synthetic.jsonl

if __name__ == "__main__":
    print("Placeholder: generate_sft.py - Generate synthetic SFT pairs")

