#!/usr/bin/env python3
"""
Convert Hugging Face Polish dataset to JSONL format for Granite fine-tuning.

Dataset: Lajonbot/alpaca-dolly-chrisociepa-instruction-only-polish
Format: {"instruction": "...", "input": "...", "output": "..."}
"""

import json
from datasets import load_dataset

def convert_dataset(output_file="polish_train.jsonl", max_samples=None):
    """
    Download and convert the Polish dataset to JSONL format.
    
    Args:
        output_file: Output JSONL file path
        max_samples: Maximum number of samples to convert (None = all)
    """
    print("Downloading dataset from Hugging Face...")
    print("Dataset: Lajonbot/alpaca-dolly-chrisociepa-instruction-only-polish")
    
    # Load the dataset
    dataset = load_dataset(
        "Lajonbot/alpaca-dolly-chrisociepa-instruction-only-polish",
        split="train"
    )
    
    print(f"✓ Dataset loaded: {len(dataset)} examples")
    
    # Limit samples if specified
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Using {len(dataset)} samples")
    
    # Convert and save
    print(f"\nConverting to JSONL format...")
    converted_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in dataset:
            # The dataset has 'instruction' and 'input' fields
            # We'll use 'input' as the output since it contains the response
            converted_example = {
                "instruction": example["instruction"],
                "input": "",  # Empty input field
                "output": example["input"]  # The 'input' field contains the actual output/response
            }
            
            f.write(json.dumps(converted_example, ensure_ascii=False) + '\n')
            converted_count += 1
            
            # Progress indicator
            if converted_count % 1000 == 0:
                print(f"  Converted {converted_count}/{len(dataset)} examples...")
    
    print(f"\n✓ Conversion complete!")
    print(f"  Output file: {output_file}")
    print(f"  Total examples: {converted_count}")
    print(f"\nSample example:")
    print("="*60)
    
    # Show a sample
    with open(output_file, 'r', encoding='utf-8') as f:
        sample = json.loads(f.readline())
        print(f"Instruction: {sample['instruction'][:100]}...")
        print(f"Input: {sample['input']}")
        print(f"Output: {sample['output'][:100]}...")
    print("="*60)
    
    return converted_count


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert Polish dataset to JSONL format for Granite fine-tuning"
    )
    parser.add_argument(
        "--output",
        default="polish_train.jsonl",
        help="Output JSONL file path (default: polish_train.jsonl)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to convert (default: all)"
    )
    
    args = parser.parse_args()
    
    try:
        convert_dataset(args.output, args.max_samples)
        print("\n✓ Ready to use with granite_tiny_polish_finetuning_pro.ipynb!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        exit(1)

# Made with Bob
