# Correct Model Names and Usage

## Available Model Names

### 1. Base Model (HuggingFace)
```python
MODEL_NAME = "ibm-granite/granite-4.0-h-tiny"
```
- Official IBM model on HuggingFace
- ~4GB download
- Works but slower

### 2. Base Model (Unsloth Optimized) ✅ RECOMMENDED
```python
MODEL_NAME = "unsloth/granite-4.0-h-tiny"
```
- Unsloth-optimized version
- ~4GB download
- 2x faster inference
- **This is what the notebook uses**

### 3. Your Fine-tuned Model (After Training)
```python
MODEL_NAME = "/content/drive/MyDrive/granite4-tiny-h-polish-lora-pro"
```
- Only exists AFTER you complete fine-tuning
- Saved to Google Drive during training
- This is what's failing in your case

## Your Current Error Explained

You're trying to load:
```python
MODEL_NAME = "/content/drive/MyDrive/granite4-tiny-h-polish-lora-pro"
```

This path doesn't exist because you haven't fine-tuned the model yet!

## Solution: Test Base Model First

**In Step 5 of the notebook, use Option A instead of Option B:**

```python
# Option A: Base model (downloads automatically from HuggingFace)
MODEL_NAME = "unsloth/granite-4.0-h-tiny"
OUTPUT_FILE = "results_before_finetuning.json"

print(f"Testing: {MODEL_NAME}")
print(f"This model will be downloaded from HuggingFace (~4GB)")
print(f"Results will be saved to: {OUTPUT_FILE}")
```

This will:
1. ✅ Download the base model automatically (~4GB, takes 2-3 minutes)
2. ✅ Test it on Polish questions
3. ✅ Save results to `results_before_finetuning.json`
4. ✅ Give you a baseline to compare against later

## Complete Workflow

### Phase 1: Test Base Model (What You Should Do Now)
```python
# In notebook Step 5, use:
MODEL_NAME = "unsloth/granite-4.0-h-tiny"
OUTPUT_FILE = "results_before_finetuning.json"
```

Run the notebook → Download `results_before_finetuning.json`

### Phase 2: Fine-tune Model (Do This Next)
1. Open `03-granite_tiny_polish_finetuning_pro.ipynb`
2. Upload `02-polish_train.jsonl`
3. Run all cells (takes 1-2 hours)
4. Save model to Google Drive as `granite4-tiny-h-polish-lora-pro`

### Phase 3: Test Fine-tuned Model (After Training)
```python
# In notebook Step 5, use:
from google.colab import drive
drive.mount('/content/drive')

MODEL_NAME = "/content/drive/MyDrive/granite4-tiny-h-polish-lora-pro"
OUTPUT_FILE = "results_after_finetuning.json"
```

Run the notebook → Download `results_after_finetuning.json`

### Phase 4: Compare Results
Compare `results_before_finetuning.json` vs `results_after_finetuning.json` to see improvement!

## Quick Fix for Your Current Error

**Replace the Step 5 Option B cell with this:**

```python
# Test base model first (no fine-tuning needed)
MODEL_NAME = "unsloth/granite-4.0-h-tiny"
OUTPUT_FILE = "results_before_finetuning.json"

print(f"Testing: {MODEL_NAME}")
print(f"This model will be downloaded from HuggingFace (~4GB)")
print(f"Results will be saved to: {OUTPUT_FILE}")
print("\n✓ No Google Drive needed - model downloads automatically")
```

Then continue with Step 6 and the model will download automatically.

## Summary

- ✅ **`unsloth/granite-4.0-h-tiny`** = Base model (use this now)
- ❌ **`/content/drive/MyDrive/granite4-tiny-h-polish-lora-pro`** = Fine-tuned model (doesn't exist yet)

You haven't lost any model link - you just need to test the base model first before fine-tuning!