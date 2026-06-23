# Fix: "No config file found" Error at Step 6

## Error Message
```
RuntimeError: Unsloth: No config file found - are you sure the `model_name` is correct?
If you're using a model on your local device, confirm if the folder location exists.
```

## Root Cause

The model path `/content/drive/MyDrive/granite4-tiny-h-polish-lora-pro` either:
1. **Doesn't exist** - Model wasn't saved during fine-tuning
2. **Wrong path** - Model saved to different location
3. **Incomplete save** - Missing required files (config.json, adapter_config.json)
4. **Drive not mounted** - Google Drive not properly connected

## Solution 1: Verify Model Exists

Add this cell **BEFORE Step 6** to check if the model exists:

```python
import os

# Check if Drive is mounted
if not os.path.exists('/content/drive'):
    print("❌ Google Drive not mounted!")
    print("Run the Drive mount cell first (Option B in Step 5)")
else:
    print("✓ Google Drive is mounted")

# Check if model folder exists
model_path = "/content/drive/MyDrive/granite4-tiny-h-polish-lora-pro"
if os.path.exists(model_path):
    print(f"✓ Model folder exists: {model_path}")
    
    # List files in model folder
    files = os.listdir(model_path)
    print(f"\nFiles in model folder ({len(files)} files):")
    for f in sorted(files)[:20]:  # Show first 20 files
        print(f"  - {f}")
    
    # Check for required files
    required_files = ['config.json', 'adapter_config.json', 'adapter_model.safetensors']
    missing = [f for f in required_files if f not in files]
    
    if missing:
        print(f"\n⚠ Missing required files: {missing}")
    else:
        print("\n✓ All required files present")
else:
    print(f"❌ Model folder NOT found: {model_path}")
    print("\nPossible locations to check:")
    
    # Check common alternative locations
    base_dir = "/content/drive/MyDrive"
    if os.path.exists(base_dir):
        print(f"\nContents of {base_dir}:")
        for item in os.listdir(base_dir):
            if 'granite' in item.lower() or 'polish' in item.lower():
                print(f"  - {item}")
```

## Solution 2: Find Your Model

If the model exists but in a different location:

```python
import os
from pathlib import Path

# Search for granite models in Google Drive
drive_path = Path("/content/drive/MyDrive")
print("Searching for granite models in Google Drive...\n")

found_models = []
for root, dirs, files in os.walk(drive_path):
    # Look for folders containing granite model files
    if 'config.json' in files or 'adapter_config.json' in files:
        if 'granite' in root.lower() or 'polish' in root.lower():
            found_models.append(root)
            print(f"✓ Found model: {root}")

if found_models:
    print(f"\n✓ Found {len(found_models)} model(s)")
    print("\nUpdate MODEL_NAME to one of these paths:")
    for model in found_models:
        print(f'  MODEL_NAME = "{model}"')
else:
    print("\n❌ No granite models found in Google Drive")
    print("You need to fine-tune the model first using:")
    print("  03-granite_tiny_polish_finetuning_pro.ipynb")
```

## Solution 3: Test Base Model Instead

If you haven't fine-tuned yet, test the base model first:

**Replace Step 5 Option B with Option A:**

```python
# Option A: Base model (downloads automatically from HuggingFace)
MODEL_NAME = "unsloth/granite-4.0-h-tiny"
OUTPUT_FILE = "results_before_finetuning.json"

print(f"Testing: {MODEL_NAME}")
print(f"This model will be downloaded from HuggingFace (~4GB)")
print(f"Results will be saved to: {OUTPUT_FILE}")
```

This will download the base model automatically - no Google Drive needed.

## Solution 4: Re-save Model with Correct Path

If you have the model but it's not loading, re-save it properly:

```python
# In your fine-tuning notebook, use this save command:
model.save_pretrained_merged(
    "granite4-tiny-h-polish-lora-pro",
    tokenizer,
    save_method="merged_16bit",  # or "lora" for LoRA adapters only
)

# Then copy to Google Drive
import shutil
shutil.copytree(
    "granite4-tiny-h-polish-lora-pro",
    "/content/drive/MyDrive/granite4-tiny-h-polish-lora-pro",
    dirs_exist_ok=True
)
print("✓ Model saved to Google Drive")
```

## Solution 5: Mount Google Drive Properly

If Drive isn't mounted, add this cell before Step 5 Option B:

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Verify mount
import os
if os.path.exists('/content/drive/MyDrive'):
    print("✓ Google Drive mounted successfully")
    print(f"Contents: {os.listdir('/content/drive/MyDrive')[:10]}")
else:
    print("❌ Google Drive mount failed")
```

## Quick Diagnostic Checklist

Run this complete diagnostic:

```python
import os
from pathlib import Path

print("="*60)
print("MODEL LOADING DIAGNOSTIC")
print("="*60)

# 1. Check Drive mount
drive_mounted = os.path.exists('/content/drive/MyDrive')
print(f"\n1. Google Drive mounted: {'✓ YES' if drive_mounted else '❌ NO'}")

# 2. Check model path
model_path = "/content/drive/MyDrive/granite4-tiny-h-polish-lora-pro"
model_exists = os.path.exists(model_path)
print(f"2. Model folder exists: {'✓ YES' if model_exists else '❌ NO'}")

if model_exists:
    # 3. Check files
    files = os.listdir(model_path)
    print(f"3. Files in model folder: {len(files)}")
    
    # 4. Check required files
    required = ['config.json', 'adapter_config.json']
    has_config = any(f in files for f in required)
    print(f"4. Has config files: {'✓ YES' if has_config else '❌ NO'}")
    
    if has_config:
        print("\n✓ Model should load correctly!")
        print(f"Use: MODEL_NAME = '{model_path}'")
    else:
        print("\n❌ Model folder incomplete - missing config files")
        print("Re-save the model from fine-tuning notebook")
else:
    print("\n❌ Model not found at expected location")
    print("\nOptions:")
    print("1. Fine-tune model first using 03-granite_tiny_polish_finetuning_pro.ipynb")
    print("2. Test base model instead: MODEL_NAME = 'unsloth/granite-4.0-h-tiny'")
    print("3. Check if model saved to different location (run search above)")

print("="*60)
```

## Recommended Workflow

1. **First time?** → Use base model (Option A in Step 5)
2. **After fine-tuning?** → Run diagnostic above to find model
3. **Model not found?** → Fine-tune first, then test
4. **Model incomplete?** → Re-save from fine-tuning notebook

## Summary

The error means the model path is wrong or the model doesn't exist. Use the diagnostic script above to identify the exact issue, then apply the appropriate solution.