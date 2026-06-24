# Converting Polish Dataset for Granite Fine-Tuning

This guide explains how to download and convert the Polish instruction dataset from Hugging Face for use with the Granite fine-tuning notebooks.

## Dataset Information

- **Source**: [Lajonbot/alpaca-dolly-chrisociepa-instruction-only-polish](https://huggingface.co/datasets/Lajonbot/alpaca-dolly-chrisociepa-instruction-only-polish)
- **Size**: ~48.6k examples
- **Language**: Polish
- **Format**: Instruction-following dataset

## Quick Start

### Option 1: Using the Conversion Script (Recommended)

1. **Install dependencies**:
```bash
pip install datasets
```

2. **Run the conversion script**:
```bash
python convert_polish_dataset.py
```

This will:
- Download the dataset from Hugging Face
- Convert it to JSONL format
- Save as `polish_train.jsonl`

3. **Use with notebook**: Upload `polish_train.jsonl` to your Colab notebook

### Option 2: Convert a Subset

To convert only a subset (e.g., 1000 examples for testing):

```bash
python convert_polish_dataset.py --max-samples 1000
```

### Option 3: Custom Output File

```bash
python convert_polish_dataset.py --output my_polish_data.jsonl
```

## Manual Conversion (In Colab)

If you prefer to convert the dataset directly in your Colab notebook, add this cell:

```python
# Install datasets library
!pip install datasets

# Download and convert
from datasets import load_dataset
import json

print("Downloading Polish dataset...")
dataset = load_dataset(
    "Lajonbot/alpaca-dolly-chrisociepa-instruction-only-polish",
    split="train"
)

print(f"Dataset loaded: {len(dataset)} examples")

# Convert to JSONL format
print("Converting to JSONL...")
with open('polish_train.jsonl', 'w', encoding='utf-8') as f:
    for example in dataset:
        converted = {
            "instruction": example["instruction"],
            "input": "",
            "output": example["input"]  # Note: 'input' field contains the response
        }
        f.write(json.dumps(converted, ensure_ascii=False) + '\n')

print("✓ Conversion complete!")
print(f"Created: polish_train.jsonl with {len(dataset)} examples")

# Preview
!head -n 1 polish_train.jsonl
```

## Dataset Format

### Original Format (Hugging Face)
```json
{
  "instruction": "Wyjaśnij, dlaczego...",
  "input": "Ponieważ jest to ciąg..."
}
```

### Converted Format (for Granite)
```json
{
  "instruction": "Wyjaśnij, dlaczego...",
  "input": "",
  "output": "Ponieważ jest to ciąg..."
}
```

**Note**: The original dataset's `input` field contains the actual response/output, so we map it to the `output` field in our format.

## Usage in Notebooks

After conversion, you can use the dataset in either notebook:

1. **granite_tiny_polish_finetuning.ipynb** (Free tier)
2. **granite_tiny_polish_finetuning_pro.ipynb** (Colab Pro)

Simply upload `polish_train.jsonl` when prompted in Step 3 of the notebook.

## Dataset Statistics

- **Total examples**: ~48,600
- **Average instruction length**: ~100 characters
- **Average output length**: ~90 characters
- **Language**: Polish
- **Task types**: Various instruction-following tasks

## Troubleshooting

### Issue: "Import datasets could not be resolved"
**Solution**: Install the datasets library:
```bash
pip install datasets
```

### Issue: "Out of memory"
**Solution**: Convert a smaller subset:
```bash
python convert_polish_dataset.py --max-samples 5000
```

### Issue: "File not found in Colab"
**Solution**: Make sure to upload the generated `polish_train.jsonl` file to Colab using the file upload option in Step 3.

## Next Steps

1. Convert the dataset using one of the methods above
2. Open your chosen notebook (standard or pro)
3. Upload `polish_train.jsonl` in Step 3
4. Continue with the fine-tuning process

## Additional Resources

- [Original Dataset on Hugging Face](https://huggingface.co/datasets/Lajonbot/alpaca-dolly-chrisociepa-instruction-only-polish)
- [Granite Fine-Tuning Guide](GRANITE_TINY_FINETUNING_GUIDE.md)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)