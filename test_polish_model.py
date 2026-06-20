#!/usr/bin/env python3
"""
Test script for evaluating Granite Tiny model on Polish tasks.
Tests the model before and after fine-tuning to measure improvement.

Usage:
    python test_polish_model.py --model unsloth/granite-4.0-h-tiny --output results_before.json
    python test_polish_model.py --model granite4-tiny-h-polish-lora-pro --output results_after.json
"""

import json
import argparse
import torch
from datetime import datetime
from unsloth import FastLanguageModel


def load_test_questions(filepath="polish_test_questions.jsonl"):
    """Load test questions from JSONL file."""
    questions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


def load_model(model_path, max_seq_length=2048):
    """Load the model and tokenizer."""
    print(f"\nLoading model: {model_path}")
    print("This may take a few minutes...\n")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    print("✓ Model loaded successfully!\n")
    return model, tokenizer


def format_prompt(instruction, input_text=""):
    """Format prompt using Granite's chat template."""
    user_message = instruction
    if input_text:
        user_message += f"\n\n{input_text}"
    
    prompt = f"""<|start_of_role|>user<|end_of_role|>
{user_message}
<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>
"""
    return prompt


def generate_response(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7):
    """Generate response from the model."""
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract assistant response
    assistant_start = response.find("<|start_of_role|>assistant<|end_of_role|>")
    if assistant_start != -1:
        assistant_response = response[assistant_start + len("<|start_of_role|>assistant<|end_of_role|>"):].strip()
        # Remove end token if present
        assistant_response = assistant_response.replace("<|end_of_text|>", "").strip()
        return assistant_response
    
    return response


def test_model(model, tokenizer, questions, max_new_tokens=512, temperature=0.7):
    """Test model on all questions."""
    results = []
    
    print(f"Testing model on {len(questions)} questions...")
    print("=" * 60)
    
    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Category: {q['category']}")
        print(f"Question: {q['instruction']}")
        if q['input']:
            print(f"Input: {q['input'][:100]}...")
        
        # Format prompt
        prompt = format_prompt(q['instruction'], q['input'])
        
        # Generate response
        try:
            response = generate_response(
                model, tokenizer, prompt, 
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            
            print(f"Response: {response[:200]}...")
            
            result = {
                "id": q['id'],
                "category": q['category'],
                "instruction": q['instruction'],
                "input": q['input'],
                "expected_output": q['expected_output'],
                "model_output": response,
                "success": True
            }
        except Exception as e:
            print(f"Error: {str(e)}")
            result = {
                "id": q['id'],
                "category": q['category'],
                "instruction": q['instruction'],
                "input": q['input'],
                "expected_output": q['expected_output'],
                "model_output": f"ERROR: {str(e)}",
                "success": False
            }
        
        results.append(result)
    
    print("\n" + "=" * 60)
    print(f"✓ Testing complete! {len(results)} questions processed.")
    
    return results


def save_results(results, output_file, model_name):
    """Save test results to JSON file."""
    output = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(results),
        "successful": sum(1 for r in results if r['success']),
        "failed": sum(1 for r in results if not r['success']),
        "results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    print(f"  Total questions: {output['total_questions']}")
    print(f"  Successful: {output['successful']}")
    print(f"  Failed: {output['failed']}")


def print_summary(results):
    """Print summary of results by category."""
    print("\n" + "=" * 60)
    print("SUMMARY BY CATEGORY")
    print("=" * 60)
    
    categories = {}
    for r in results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = {'total': 0, 'success': 0}
        categories[cat]['total'] += 1
        if r['success']:
            categories[cat]['success'] += 1
    
    for cat, stats in sorted(categories.items()):
        success_rate = (stats['success'] / stats['total']) * 100
        print(f"{cat:20s}: {stats['success']}/{stats['total']} ({success_rate:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description='Test Granite Tiny model on Polish tasks')
    parser.add_argument('--model', type=str, required=True,
                       help='Model path (e.g., unsloth/granite-4.0-h-tiny or granite4-tiny-h-polish-lora-pro)')
    parser.add_argument('--questions', type=str, default='polish_test_questions.jsonl',
                       help='Path to test questions file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file for results')
    parser.add_argument('--max-tokens', type=int, default=512,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature')
    parser.add_argument('--max-seq-length', type=int, default=2048,
                       help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("⚠ Warning: CUDA not available. This script requires a GPU.")
        return
    
    print("=" * 60)
    print("GRANITE TINY POLISH TEST SUITE")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Questions: {args.questions}")
    print(f"Output: {args.output}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    
    # Load test questions
    questions = load_test_questions(args.questions)
    print(f"\n✓ Loaded {len(questions)} test questions")
    
    # Load model
    model, tokenizer = load_model(args.model, args.max_seq_length)
    
    # Test model
    results = test_model(
        model, tokenizer, questions,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    # Print summary
    print_summary(results)
    
    # Save results
    save_results(results, args.output, args.model)
    
    print("\n" + "=" * 60)
    print("✓ Testing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

# Made with Bob
