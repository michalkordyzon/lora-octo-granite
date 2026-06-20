# Polish Model Testing Suite

This directory contains a comprehensive test suite to evaluate Granite Tiny model's Polish language capabilities before and after fine-tuning.

## Files

- **`polish_test_questions.jsonl`** - 20 diverse test questions covering different categories
- **`test_polish_model.py`** - Python script to run tests and generate results
- **`README_TESTING.md`** - This file

## Test Categories

The test suite includes 20 questions across these categories:

1. **Grammar** - Error correction
2. **Translation** - English to Polish
3. **Summarization** - Text condensation
4. **Math** - Word problems
5. **Classification** - Sentiment/tone analysis
6. **Question Answering** - Reading comprehension
7. **Creative** - Poetry generation
8. **Reasoning** - Logic puzzles
9. **Formatting** - Text structure
10. **Extraction** - Information retrieval
11. **Comparison** - Concept analysis
12. **Instruction Following** - Precise task execution
13. **Sentiment** - Emotion scoring
14. **Paraphrasing** - Rewording
15. **Entity Recognition** - Named entity extraction
16. **Definition** - Concept explanation
17. **Advice** - Practical recommendations
18. **Error Detection** - Finding mistakes
19. **Continuation** - Story completion
20. **Factual** - Knowledge questions

## Usage

### Step 1: Test Base Model (Before Fine-tuning)

```bash
python test_polish_model.py \
  --model unsloth/granite-4.0-h-tiny \
  --output results_before.json
```

### Step 2: Fine-tune the Model

Follow the instructions in `granite_tiny_polish_finetuning_pro.ipynb` to fine-tune the model on your Polish dataset.

### Step 3: Test Fine-tuned Model (After Fine-tuning)

```bash
python test_polish_model.py \
  --model granite4-tiny-h-polish-lora-pro \
  --output results_after.json
```

### Step 4: Compare Results

```bash
python compare_results.py \
  --before results_before.json \
  --after results_after.json \
  --output comparison_report.md
```

## Command Line Options

```
--model MODEL_PATH          Path to model (required)
--questions QUESTIONS_FILE  Path to test questions (default: polish_test_questions.jsonl)
--output OUTPUT_FILE        Output JSON file for results (required)
--max-tokens MAX_TOKENS     Maximum tokens to generate (default: 512)
--temperature TEMPERATURE   Sampling temperature (default: 0.7)
--max-seq-length LENGTH     Maximum sequence length (default: 2048)
```

## Output Format

The test script generates a JSON file with:

```json
{
  "model": "model_name",
  "timestamp": "2026-06-20T18:00:00",
  "total_questions": 20,
  "successful": 20,
  "failed": 0,
  "results": [
    {
      "id": 1,
      "category": "grammar",
      "instruction": "...",
      "input": "...",
      "expected_output": "...",
      "model_output": "...",
      "success": true
    }
  ]
}
```

## Evaluation Metrics

When comparing before/after results, consider:

1. **Accuracy** - How well does the output match expected results?
2. **Fluency** - Is the Polish natural and grammatically correct?
3. **Completeness** - Does it answer the full question?
4. **Relevance** - Does it stay on topic?
5. **Format** - Does it follow formatting instructions?

## Example Workflow

```bash
# 1. Test base model
python test_polish_model.py \
  --model unsloth/granite-4.0-h-tiny \
  --output results_before.json

# 2. Fine-tune (in Colab notebook)
# Upload polish_train.jsonl and run granite_tiny_polish_finetuning_pro.ipynb

# 3. Download fine-tuned model from Colab
# Place in: granite4-tiny-h-polish-lora-pro/

# 4. Test fine-tuned model
python test_polish_model.py \
  --model granite4-tiny-h-polish-lora-pro \
  --output results_after.json

# 5. Compare results manually or create comparison script
```

## Tips

- **GPU Required**: This script requires a CUDA-capable GPU
- **Memory**: Ensure you have at least 8GB GPU memory
- **Time**: Testing takes ~5-10 minutes for 20 questions
- **Temperature**: Lower temperature (0.3-0.5) for more deterministic outputs
- **Max Tokens**: Increase if responses are cut off

## Interpreting Results

### Good Signs After Fine-tuning:
- ✓ More natural Polish phrasing
- ✓ Better understanding of Polish grammar
- ✓ More accurate responses to Polish-specific questions
- ✓ Improved instruction following
- ✓ Better handling of Polish cultural context

### Potential Issues:
- ⚠ Overfitting (perfect on training data, poor on test)
- ⚠ Catastrophic forgetting (worse on general tasks)
- ⚠ Hallucinations (making up information)
- ⚠ Format inconsistencies

## Next Steps

1. **Analyze Results**: Review both JSON files to identify improvements
2. **Iterate**: If results are poor, adjust training parameters
3. **Expand Tests**: Add more test cases for specific use cases
4. **Deploy**: If satisfied, deploy the fine-tuned model

## Requirements

```bash
pip install unsloth torch transformers datasets
```

## Notes

- Test questions are NOT part of the training data
- Questions cover diverse Polish language tasks
- Expected outputs are reference answers (not strict requirements)
- Manual review is recommended for qualitative assessment