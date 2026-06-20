# Granite 4.0 H-Tiny Fine-Tuning Guide for Google Colab

## Executive Summary

This guide provides step-by-step instructions for fine-tuning IBM's Granite 4.0 H-Tiny model using LoRA/QLoRA on Google Colab, specifically optimized for Polish language tasks.

**Key Points:**
- Model: `ibm-granite/granite-4.0-h-tiny` (4B parameters)
- Method: LoRA (Low-Rank Adaptation) via Unsloth
- Platform: Google Colab (Free T4 or Pro A100)
- Target: Polish enterprise document extraction
- Training Time: 2-4 hours on T4, 30-60 minutes on A100

---

## Resource Requirements Assessment

### Model Specifications
- **Base Model Size**: ~4B parameters
- **Model Storage**: ~8GB (FP16) / ~4GB (4-bit quantized)
- **Context Length**: 2048 tokens (configurable up to 8192)

### Google Colab GPU Options

#### Option 1: Free Tier (T4 GPU)
- **GPU Memory**: 15GB VRAM
- **RAM**: 12-13GB system RAM
- **Disk**: 78GB temporary storage
- **Cost**: FREE
- **Limitations**: 
  - Session timeout after 12 hours
  - May disconnect after 90 minutes of inactivity
  - GPU availability not guaranteed
- **Recommended for**: Proof of concept (200-500 examples)

#### Option 2: Colab Pro (T4/V100)
- **GPU Memory**: 15-16GB VRAM
- **RAM**: 25-27GB system RAM
- **Disk**: 166GB temporary storage
- **Cost**: ~$10/month
- **Benefits**: 
  - Longer sessions (24 hours)
  - Better GPU availability
  - Background execution
- **Recommended for**: Small to medium datasets (2,000-5,000 examples)

#### Option 3: Colab Pro+ (A100)
- **GPU Memory**: 40GB VRAM
- **RAM**: 51GB system RAM
- **Disk**: 166GB+ temporary storage
- **Cost**: ~$50/month
- **Benefits**: 
  - Fastest training (5-10x faster than T4)
  - Can handle larger batches
  - Extended sessions
- **Recommended for**: Production models (10,000-50,000 examples)

### Memory Usage Breakdown (T4 with 4-bit quantization)

```
Component                    Memory Usage
─────────────────────────────────────────
Base Model (4-bit)           ~4.0 GB
LoRA Adapters                ~0.5 GB
Optimizer States             ~2.0 GB
Gradient Checkpointing       ~1.5 GB
Batch Processing (bs=1)      ~1.0 GB
System Overhead              ~1.0 GB
─────────────────────────────────────────
Total Estimated              ~10.0 GB
Available on T4              15.0 GB
Safety Margin                ~5.0 GB ✓
```

**Verdict**: T4 GPU is sufficient for this fine-tuning task.

---

## Dataset Preparation

### Recommended Dataset Sizes

| Goal                          | Examples      | Training Time (T4) | Training Time (A100) |
|-------------------------------|---------------|--------------------|--------------------|
| Proof of concept              | 200-500       | 30-60 min          | 5-10 min           |
| Useful internal model         | 2,000-5,000   | 2-4 hours          | 20-40 min          |
| Strong narrow model           | 10,000-50,000 | 8-20 hours         | 1-3 hours          |

### Data Format

Create a JSONL file (`polish_train.jsonl`) with this structure:

```json
{"instruction": "Wyodrębnij dane z tekstu i zwróć JSON.", "input": "Faktura nr FV/123/2026 z dnia 12.06.2026 na kwotę 4500 PLN dla firmy ABC Sp. z o.o.", "output": "{\"typ\":\"faktura\",\"numer\":\"FV/123/2026\",\"data\":\"2026-06-12\",\"kwota\":\"4500 PLN\",\"firma\":\"ABC Sp. z o.o.\"}"}
{"instruction": "Sklasyfikuj wiadomość.", "input": "Klient prosi o pilny kontakt w sprawie błędu produkcyjnego.", "output": "pilne / wymaga odpowiedzi"}
{"instruction": "Podsumuj tekst po polsku w 3 punktach.", "input": "Długi fragment dokumentu...", "output": "- Punkt 1\n- Punkt 2\n- Punkt 3"}
```

### Example Use Cases for Polish Fine-Tuning

1. **Polish document classification**
2. **Polish metadata extraction**
3. **Polish RAG answer generation**
4. **Polish customer support replies**
5. **Polish procurement document analysis**
6. **Polish summarization into structured bullets**
7. **Polish SQL/query explanation**
8. **Polish email/action-item extraction**

### Public Polish Datasets (Optional)

- `Lajonbot/alpaca-dolly-chrisociepa-instruction-only-polish` (~48.6k examples)
- SpeakLeash/Bielik project datasets
- PLLuM Polish language model datasets

**Note**: Your own task-specific data will yield better results than generic datasets.

---

## Step-by-Step Colab Setup

### Step 1: Check GPU Availability

```python
# Check GPU type and memory
!nvidia-smi

# Expected output for T4:
# Tesla T4, 15GB VRAM
```

### Step 2: Install Dependencies

```python
# Install Unsloth and dependencies
!pip install --upgrade --no-cache-dir unsloth
!pip install --upgrade datasets trl transformers accelerate bitsandbytes

# Verify installation
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

**Installation time**: 3-5 minutes

### Step 3: Upload Your Dataset

```python
from google.colab import files
import os

# Option A: Upload from local machine
uploaded = files.upload()  # Upload polish_train.jsonl

# Option B: Download from Google Drive
from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/polish_train.jsonl .

# Option C: Download from URL
!wget https://your-url.com/polish_train.jsonl

# Verify file
!wc -l polish_train.jsonl
!head -n 3 polish_train.jsonl
```

### Step 4: Load Model with Unsloth

```python
from unsloth import FastLanguageModel
import torch

# Configuration
max_seq_length = 2048  # Can increase to 4096 or 8192 if needed
dtype = None  # Auto-detect (Float16 for T4/V100, BFloat16 for A100)
load_in_4bit = True  # Use 4-bit quantization to save memory

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/granite-4.0-h-tiny",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

print(f"Model loaded successfully!")
print(f"Model size: {model.get_memory_footprint() / 1e9:.2f} GB")
```

**Loading time**: 2-3 minutes

### Step 5: Configure LoRA

```python
# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                      # LoRA rank (8, 16, 32, or 64)
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=32,             # LoRA alpha (typically 2*r)
    lora_dropout=0.05,         # Dropout for regularization
    bias="none",               # Bias training strategy
    use_gradient_checkpointing="unsloth",  # Memory optimization
    random_state=42,
    use_rslora=False,          # Rank-stabilized LoRA
    loftq_config=None,         # LoftQ quantization
)

# Print trainable parameters
model.print_trainable_parameters()
# Expected: ~0.5-1% of total parameters are trainable
```

**LoRA Configuration Explained:**
- `r=16`: Rank of LoRA matrices (higher = more capacity, more memory)
- `lora_alpha=32`: Scaling factor (typically 2*r)
- `lora_dropout=0.05`: Prevents overfitting
- `use_gradient_checkpointing`: Reduces memory usage by ~40%

### Step 6: Prepare Dataset

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("json", data_files="polish_train.jsonl", split="train")

print(f"Dataset size: {len(dataset)} examples")
print(f"Sample example: {dataset[0]}")

# Format examples for Granite chat template
def format_example(example):
    """
    Format examples using Granite's chat template:
    <|start_of_role|>user<|end_of_role|>
    {instruction}
    {input}
    <|end_of_text|>
    <|start_of_role|>assistant<|end_of_role|>
    {output}
    <|end_of_text|>
    """
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]
    
    # Combine instruction and input
    user_message = instruction
    if input_text:
        user_message += f"\n\n{input_text}"
    
    # Format with chat template
    text = f"""<|start_of_role|>user<|end_of_role|>
{user_message}
<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>
{output}
<|end_of_text|>"""
    
    return {"text": text}

# Apply formatting
dataset = dataset.map(format_example, remove_columns=dataset.column_names)

# Verify formatting
print("\nFormatted example:")
print(dataset[0]["text"])
```

### Step 7: Configure Training

```python
from transformers import TrainingArguments
from trl import SFTTrainer

# Training configuration
training_args = TrainingArguments(
    # Output
    output_dir="./granite4-tiny-h-polish-lora",
    
    # Batch size and accumulation
    per_device_train_batch_size=1,      # Batch size per GPU
    gradient_accumulation_steps=8,       # Effective batch size = 1 * 8 = 8
    
    # Learning rate and schedule
    learning_rate=2e-4,                  # LoRA typically uses 1e-4 to 5e-4
    lr_scheduler_type="cosine",          # Cosine decay
    warmup_steps=50,                     # Warmup steps
    
    # Training duration
    num_train_epochs=2,                  # 2-3 epochs typical for LoRA
    max_steps=-1,                        # -1 means use num_train_epochs
    
    # Optimization
    optim="adamw_8bit",                  # 8-bit AdamW saves memory
    weight_decay=0.01,                   # L2 regularization
    max_grad_norm=1.0,                   # Gradient clipping
    
    # Precision
    fp16=not torch.cuda.is_bf16_supported(),  # FP16 for T4/V100
    bf16=torch.cuda.is_bf16_supported(),      # BF16 for A100
    
    # Logging and saving
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,                  # Keep only 2 checkpoints
    
    # Evaluation (optional)
    # eval_steps=100,
    # evaluation_strategy="steps",
    
    # Other
    report_to="none",                    # Disable wandb/tensorboard
    seed=42,
)

# Create trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=training_args,
    packing=False,  # Don't pack multiple examples into one sequence
)
```

**Training Configuration Explained:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `per_device_train_batch_size` | 1 | Batch size per GPU (limited by memory) |
| `gradient_accumulation_steps` | 8 | Accumulate gradients over 8 steps (effective batch size = 8) |
| `learning_rate` | 2e-4 | Higher than full fine-tuning (1e-5) because LoRA is more stable |
| `num_train_epochs` | 2 | 2-3 epochs typical for LoRA (more can cause overfitting) |
| `optim` | adamw_8bit | 8-bit optimizer saves ~50% memory |
| `fp16/bf16` | Auto | Mixed precision training (2x faster, 50% less memory) |

### Step 8: Train the Model

```python
# Start training
print("Starting training...")
print(f"Total examples: {len(dataset)}")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"Total steps: {len(dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}")

# Train
trainer.train()

print("\nTraining complete!")
```

**Expected Training Times:**

| Dataset Size | GPU Type | Time Estimate |
|--------------|----------|---------------|
| 500 examples | T4       | 30-60 min     |
| 2,000 examples | T4     | 2-3 hours     |
| 5,000 examples | T4     | 4-6 hours     |
| 500 examples | A100     | 5-10 min      |
| 2,000 examples | A100   | 20-30 min     |
| 5,000 examples | A100   | 40-60 min     |

### Step 9: Save the Model

```python
# Save LoRA adapter
model.save_pretrained("granite4-tiny-h-polish-lora")
tokenizer.save_pretrained("granite4-tiny-h-polish-lora")

# Save to Google Drive (recommended)
from google.colab import drive
drive.mount('/content/drive')

!cp -r granite4-tiny-h-polish-lora /content/drive/MyDrive/

print("Model saved successfully!")
```

**Saved Files:**
- `adapter_config.json` - LoRA configuration
- `adapter_model.safetensors` - LoRA weights (~50-200MB)
- `tokenizer.json` - Tokenizer configuration
- `tokenizer_config.json` - Tokenizer settings

### Step 10: Test the Fine-Tuned Model

```python
# Enable inference mode
FastLanguageModel.for_inference(model)

# Test function
def test_model(instruction, input_text=""):
    """Test the fine-tuned model"""
    # Format prompt
    user_message = instruction
    if input_text:
        user_message += f"\n\n{input_text}"
    
    prompt = f"""<|start_of_role|>user<|end_of_role|>
{user_message}
<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>
"""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
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

# Test examples
print("Test 1: Document extraction")
result1 = test_model(
    "Wyodrębnij dane z tekstu i zwróć JSON.",
    "Faktura nr FV/456/2026 z dnia 15.06.2026 na kwotę 7800 PLN dla firmy XYZ Sp. z o.o."
)
print(result1)
print("\n" + "="*50 + "\n")

print("Test 2: Classification")
result2 = test_model(
    "Sklasyfikuj wiadomość.",
    "Klient dziękuje za szybką realizację zamówienia."
)
print(result2)
print("\n" + "="*50 + "\n")

print("Test 3: Summarization")
result3 = test_model(
    "Podsumuj tekst po polsku w 3 punktach.",
    "Projekt modernizacji infrastruktury IT obejmuje migrację do chmury, wdrożenie nowych systemów bezpieczeństwa oraz szkolenie pracowników. Budżet wynosi 500 tys. PLN, a termin realizacji to 6 miesięcy."
)
print(result3)
```

---

## Evaluation Framework

### Create Test Dataset

```python
# Create test set (100-300 examples, never used in training)
test_data = [
    {
        "instruction": "Wyodrębnij dane z tekstu i zwróć JSON.",
        "input": "Umowa nr UM/789/2026 z dnia 20.06.2026...",
        "expected_output": "{\"typ\":\"umowa\",\"numer\":\"UM/789/2026\",...}"
    },
    # Add more test examples...
]
```

### Evaluation Metrics

| Task Type | Metrics | Tools |
|-----------|---------|-------|
| Classification | Accuracy, F1, Precision, Recall | scikit-learn |
| Extraction | Exact match, JSON validity, Field accuracy | Custom scripts |
| Summarization | ROUGE, Human evaluation | rouge-score |
| RAG Answer | Faithfulness, Source usage | Custom evaluation |
| Polish Quality | Native speaker review | Human evaluation |
| Safety | Refusal rate, Hallucination detection | Custom tests |

### Key Evaluation Questions

1. **Does it answer in Polish naturally?**
2. **Does it keep JSON valid?**
3. **Does it follow Polish instructions correctly?**
4. **Does it hallucinate less on Polish documents?**
5. **Is it still faster and cheaper than bigger models?**

### Automated Evaluation Script

```python
import json
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(test_data):
    """Evaluate model on test dataset"""
    results = {
        "total": len(test_data),
        "correct": 0,
        "json_valid": 0,
        "responses": []
    }
    
    for example in test_data:
        # Generate prediction
        prediction = test_model(
            example["instruction"],
            example.get("input", "")
        )
        
        # Check JSON validity (if applicable)
        is_json_valid = False
        try:
            json.loads(prediction)
            is_json_valid = True
            results["json_valid"] += 1
        except:
            pass
        
        # Store result
        results["responses"].append({
            "instruction": example["instruction"],
            "input": example.get("input", ""),
            "expected": example["expected_output"],
            "predicted": prediction,
            "json_valid": is_json_valid
        })
    
    # Calculate metrics
    results["json_validity_rate"] = results["json_valid"] / results["total"]
    
    return results

# Run evaluation
# eval_results = evaluate_model(test_data)
# print(f"JSON Validity Rate: {eval_results['json_validity_rate']:.2%}")
```

---

## Deployment Options

### Option A: Serve Base Model + LoRA Adapter

**Good for**: Testing, experimentation, multiple adapters

```python
from unsloth import FastLanguageModel

# Load base model + adapter
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/granite-4.0-h-tiny",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Load LoRA adapter
model.load_adapter("granite4-tiny-h-polish-lora")

# Enable inference mode
FastLanguageModel.for_inference(model)
```

**Pros:**
- Can switch between adapters
- Smaller storage (adapter only ~50-200MB)
- Easy to update

**Cons:**
- Requires loading base model + adapter
- Slightly slower inference

### Option B: Merge Adapter and Export

**Good for**: Production deployment, simpler serving

```python
# Merge LoRA adapter into base model
model = model.merge_and_unload()

# Save merged model
model.save_pretrained("granite4-tiny-h-polish-merged")
tokenizer.save_pretrained("granite4-tiny-h-polish-merged")

# Optional: Convert to GGUF for llama.cpp/Ollama
# Requires additional tools
```

**Pros:**
- Single model file
- Faster inference
- Compatible with more serving frameworks

**Cons:**
- Larger storage (~8GB)
- Can't switch adapters

### Option C: Export to GGUF (for local deployment)

```python
# Export to GGUF format for llama.cpp/Ollama
# This requires the llama.cpp conversion tools

# 1. Save in HuggingFace format
model.save_pretrained_merged(
    "granite4-tiny-h-polish-merged",
    tokenizer,
    save_method="merged_16bit"
)

# 2. Convert to GGUF (run in terminal)
# python llama.cpp/convert.py granite4-tiny-h-polish-merged \
#   --outfile granite4-tiny-h-polish.gguf \
#   --outtype f16

# 3. Quantize (optional, for smaller size)
# ./llama.cpp/quantize granite4-tiny-h-polish.gguf \
#   granite4-tiny-h-polish-q4_k_m.gguf q4_k_m
```

**GGUF Quantization Options:**

| Format | Size | Quality | Use Case |
|--------|------|---------|----------|
| f16 | ~8GB | Best | GPU inference |
| q8_0 | ~4GB | Excellent | High-quality CPU |
| q4_k_m | ~2.5GB | Good | Balanced CPU |
| q4_0 | ~2GB | Acceptable | Low-memory CPU |

---

## Cost Analysis

### Google Colab Costs

| Tier | Monthly Cost | GPU Access | Best For |
|------|--------------|------------|----------|
| Free | $0 | T4 (limited) | POC, learning |
| Pro | ~$10 | T4/V100 (better) | Small projects |
| Pro+ | ~$50 | A100 (best) | Production |

### Training Cost Estimates

**Scenario 1: Proof of Concept (500 examples)**
- GPU: Free T4
- Time: 30-60 minutes
- Cost: $0
- Result: Basic Polish extraction capability

**Scenario 2: Internal Tool (2,000 examples)**
- GPU: Colab Pro T4
- Time: 2-3 hours
- Cost: $10/month (includes other usage)
- Result: Reliable Polish document processor

**Scenario 3: Production Model (10,000 examples)**
- GPU: Colab Pro+ A100
- Time: 1-2 hours
- Cost: $50/month (includes other usage)
- Result: High-quality Polish enterprise extractor

### Inference Cost Comparison

**Granite 4.0 H-Tiny (Fine-tuned)**
- Tokens/second: ~50-100 (T4), ~200-300 (A100)
- Cost per 1M tokens: ~$0.10-0.50 (self-hosted)
- Latency: 50-200ms

**GPT-4 (for comparison)**
- Cost per 1M tokens: ~$30-60
- Latency: 500-2000ms

**Savings**: 60-600x cheaper for narrow Polish tasks

---

## Troubleshooting

### Issue 1: Out of Memory (OOM)

**Symptoms**: CUDA out of memory error

**Solutions**:
```python
# Reduce batch size
per_device_train_batch_size=1

# Increase gradient accumulation
gradient_accumulation_steps=16

# Reduce sequence length
max_seq_length=1024

# Enable more aggressive gradient checkpointing
use_gradient_checkpointing="unsloth"

# Use 4-bit quantization
load_in_4bit=True
```

### Issue 2: Slow Training

**Symptoms**: Training takes too long

**Solutions**:
- Upgrade to Colab Pro+ (A100)
- Reduce dataset size for testing
- Increase batch size if memory allows
- Use mixed precision (fp16/bf16)

### Issue 3: Poor Polish Quality

**Symptoms**: Model doesn't respond well in Polish

**Solutions**:
- Increase training examples (aim for 2,000+)
- Improve data quality (native speaker review)
- Increase training epochs (3-4 instead of 2)
- Adjust learning rate (try 1e-4 or 3e-4)
- Use more Polish-specific examples

### Issue 4: Model Overfitting

**Symptoms**: Perfect on training data, poor on test data

**Solutions**:
```python
# Increase dropout
lora_dropout=0.1

# Add weight decay
weight_decay=0.05

# Reduce training epochs
num_train_epochs=1

# Use early stopping
evaluation_strategy="steps"
load_best_model_at_end=True
```

### Issue 5: Session Timeout

**Symptoms**: Colab disconnects during training

**Solutions**:
- Upgrade to Colab Pro (24-hour sessions)
- Save checkpoints frequently (`save_steps=100`)
- Use Google Drive for automatic backup
- Enable background execution (Pro feature)

---

## Best Practices

### 1. Data Quality Over Quantity
- Start with 200-500 high-quality examples
- Have native Polish speakers review data
- Focus on your specific use case
- Avoid generic chatbot data

### 2. Iterative Development
1. Train on 200 examples (30 min)
2. Evaluate on test set
3. Identify weaknesses
4. Add 300 more targeted examples
5. Retrain and compare
6. Repeat until satisfied

### 3. Version Control
```python
# Save each training run with version
output_dir=f"./granite4-polish-v{version}-{date}"

# Track hyperparameters
config = {
    "version": "v1.0",
    "date": "2026-06-20",
    "dataset_size": 500,
    "learning_rate": 2e-4,
    "epochs": 2,
    "lora_r": 16,
}
```

### 4. Monitor Training
```python
# Log to TensorBoard
report_to="tensorboard"

# View in Colab
%load_ext tensorboard
%tensorboard --logdir ./granite4-tiny-h-polish-lora/runs
```

### 5. Test Thoroughly
- Create diverse test cases
- Test edge cases (long documents, special characters)
- Verify JSON validity
- Check Polish grammar and naturalness
- Test refusal behavior (harmful requests)

---

## Recommended Workflow

### Week 1: Proof of Concept
1. Collect 200-500 examples
2. Train on Free Colab T4 (1 hour)
3. Evaluate basic functionality
4. Decision: Continue or pivot?

### Week 2-3: Development
1. Expand to 2,000 examples
2. Train on Colab Pro ($10)
3. Comprehensive evaluation
4. Iterate on data quality

### Week 4: Production
1. Final dataset (5,000-10,000 examples)
2. Train on Colab Pro+ A100 ($50)
3. Full evaluation suite
4. Deploy merged model

### Ongoing: Maintenance
1. Collect production feedback
2. Add edge cases to dataset
3. Retrain monthly/quarterly
4. Monitor performance metrics

---

## Example: Polish Enterprise Document Extractor

### Target Output Format

```json
{
  "document_type": "zapytanie ofertowe",
  "client": "XYZ Bank",
  "deadline": "2026-07-15",
  "required_products": [
    "watsonx.data",
    "OpenShift",
    "SQL Server integration"
  ],
  "risks": [
    "brak jasnych wymagań HA",
    "nieokreślony wolumen danych"
  ],
  "next_actions": [
    "doprecyzować RTO/RPO",
    "potwierdzić źródła danych",
    "ustalić wymagania bezpieczeństwa"
  ]
}
```

### Training Data Example

```json
{
  "instruction": "Przeanalizuj dokument i wyodrębnij kluczowe informacje w formacie JSON.",
  "input": "Zapytanie ofertowe nr ZO/2026/456\n\nKlient: XYZ Bank S.A.\nTermin składania ofert: 15 lipca 2026\n\nWymagane produkty:\n- Platforma watsonx.data do zarządzania danymi\n- Red Hat OpenShift do konteneryzacji\n- Integracja z Microsoft SQL Server\n\nUwagi:\n- Brak szczegółowych wymagań dotyczących wysokiej dostępności (HA)\n- Nie określono dokładnego wolumenu danych do przetworzenia\n\nProsimy o ofertę zawierającą:\n1. Szczegółowe wymagania RTO/RPO\n2. Potwierdzenie źródeł danych\n3. Wymagania bezpieczeństwa",
  "output": "{\"document_type\":\"zapytanie ofertowe\",\"client\":\"XYZ Bank\",\"deadline\":\"2026-07-15\",\"required_products\":[\"watsonx.data\",\"OpenShift\",\"SQL Server integration\"],\"risks\":[\"brak jasnych wymagań HA\",\"nieokreślony wolumen danych\"],\"next_actions\":[\"doprecyzować RTO/RPO\",\"potwierdzić źródła danych\",\"ustalić wymagania bezpieczeństwa\"]}"
}
```

---

## Conclusion

Fine-tuning Granite 4.0 H-Tiny for Polish tasks is:
- **Feasible**: Fits on free T4 GPU
- **Fast**: 30 minutes to 4 hours depending on dataset
- **Affordable**: $0-50/month
- **Effective**: 60-600x cheaper than GPT-4 for narrow tasks

**Key Success Factors:**
1. Focus on specific use case (not general Polish chatbot)
2. High-quality, task-specific training data
3. Iterative development with evaluation
4. Proper LoRA configuration
5. Thorough testing before deployment

**Next Steps:**
1. Define your specific Polish task
2. Collect 200-500 initial examples
3. Run proof of concept on Free Colab
4. Evaluate and iterate
5. Scale to production

---

## Additional Resources

- [IBM Granite Documentation](https://www.ibm.com/granite/docs/)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face Granite Model Card](https://huggingface.co/ibm-granite/granite-4.0-h-tiny)
- [Polish NLP Resources](https://github.com/speakleash/Bielik)

---

**Document Version**: 1.0
**Last Updated**: 2026-06-20
**Author**: AI Assistant
**License**: MIT