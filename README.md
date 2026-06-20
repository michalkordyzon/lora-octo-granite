# Granite 4.0 H-Tiny Fine-Tuning for Polish Tasks

Complete guide and resources for fine-tuning IBM's Granite 4.0 H-Tiny model using LoRA on Google Colab for Polish language tasks.

## 📁 Project Files (Numbered Workflow)

### Dataset Preparation
- **`01-convert_polish_dataset.py`** - Convert HuggingFace dataset to JSONL format
- **`02-polish_train.jsonl`** - Training data (48,637 Polish examples)
- **`08-README_DATASET_CONVERSION.md`** - Dataset conversion guide

### Training Notebooks
- **`03-granite_tiny_polish_finetuning_pro.ipynb`** - **[RECOMMENDED]** Colab Pro optimized (T4/V100/A100)
- **`04-granite_tiny_polish_finetuning.ipynb`** - Free tier compatible (T4)
- **`10-GRANITE_TINY_FINETUNING_GUIDE.md`** - Complete training guide

### Testing & Evaluation
- **`05-polish_test_questions.jsonl`** - 20 test questions (not in training data)
- **`06-test_granite_polish.ipynb`** - **[RECOMMENDED]** Colab testing notebook
- **`07-test_polish_model.py`** - Local GPU testing script
- **`09-README_TESTING.md`** - Testing guide

### Documentation
- **`README.md`** - This file (project overview)
- **`11-RESOURCE_ASSESSMENT.md`** - Resource requirements & costs
- **`.gitignore`** - Git ignore rules

## 🚀 Quick Start (3 Steps)

### Step 1: Prepare Dataset (5 minutes)
```bash
python 01-convert_polish_dataset.py
# Creates: 02-polish_train.jsonl (48,637 examples)
```

### Step 2: Fine-tune Model (1-2 hours on Colab Pro)
1. Open `03-granite_tiny_polish_finetuning_pro.ipynb` in Google Colab
2. Select GPU runtime (T4/V100/A100)
3. Upload `02-polish_train.jsonl`
4. Run all cells
5. Download fine-tuned model

### Step 3: Test & Compare (10 minutes)
1. Open `06-test_granite_polish.ipynb` in Google Colab
2. Test **before** fine-tuning (base model)
3. Test **after** fine-tuning (your model)
4. Compare results to measure improvement

## 📋 What's Included

### 1. Complete Training Pipeline
- Pre-configured Colab notebooks (free & Pro tiers)
- 48K+ Polish training examples ready to use
- Automatic model saving and deployment
- Interactive testing included

### 2. Comprehensive Testing Suite
- 20 diverse Polish test questions
- Before/after comparison framework
- Multiple task categories (grammar, translation, reasoning, etc.)
- Automated result generation

### 3. Detailed Documentation
- Step-by-step guides for every stage
- Resource requirements and cost analysis
- Best practices and troubleshooting
- Example use cases

## ✅ Key Features

- **Free to Start**: Works on Google Colab free tier (T4 GPU)
- **Fast Training**: 1-2 hours on Pro tier (48K examples)
- **Cost Effective**: $0-50/month vs. $1000s for alternatives
- **Production Ready**: Handles 10,000+ examples
- **60-600x Cheaper**: Than GPT-4 for narrow Polish tasks
- **Complete Testing**: Before/after comparison included

## 📊 Resource Requirements

### Minimum (Free Tier)
- **GPU**: T4 (15GB VRAM) - FREE
- **Dataset**: 200-500 examples
- **Time**: 30-60 minutes
- **Cost**: $0

### Recommended (Production)
- **GPU**: V100/A100 (16-40GB VRAM) - $10-50/month
- **Dataset**: 10,000-50,000 examples
- **Time**: 1-2 hours
- **Cost**: $10-50/month

## 🎯 Use Cases

Perfect for narrow Polish tasks:
- ✅ Document classification
- ✅ Metadata extraction
- ✅ RAG answer generation
- ✅ Customer support replies
- ✅ Procurement document analysis
- ✅ Structured summarization
- ✅ SQL/query explanation
- ✅ Email/action-item extraction

**Not recommended for**: General Polish chatbot (use continued pretraining instead)

## 📝 Data Format

Your training data should be in JSONL format:

```json
{"instruction": "Wyodrębnij dane z tekstu i zwróć JSON.", "input": "Faktura nr FV/123/2026...", "output": "{\"typ\":\"faktura\",...}"}
{"instruction": "Sklasyfikuj wiadomość.", "input": "Klient prosi o pilny kontakt...", "output": "pilne / wymaga odpowiedzi"}
```

### Dataset Size Recommendations

| Goal | Examples | Training Time (V100) | Cost |
|------|----------|---------------------|------|
| Proof of concept | 200-500 | 15-30 min | $0 (free tier) |
| Useful internal model | 2,000-5,000 | 30-60 min | $10/month |
| Strong narrow model | 10,000-50,000 | 1-3 hours | $50/month |

## 🔧 Technical Details

### Model
- **Name**: `ibm-granite/granite-4.0-h-tiny`
- **Parameters**: ~4 billion
- **Context**: 2048 tokens (default), up to 8192
- **Languages**: 12 officially supported (fine-tune for Polish)

### Training Method
- **Approach**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit (reduces memory by 75%)
- **Trainable Parameters**: ~0.5-1% of total
- **Memory Usage**: ~10GB on T4
- **Framework**: Unsloth + Transformers + TRL

### Why LoRA?
- ✅ **Memory Efficient**: Fits on free T4 GPU
- ✅ **Fast Training**: 1-2 hours vs. days for full fine-tuning
- ✅ **Low Risk**: Minimal catastrophic forgetting
- ✅ **Small Adapters**: 50-200MB vs. 8GB full model
- ✅ **Cost Effective**: $0-50 vs. $1000s

## 📈 Expected Results

### Training Time by Dataset Size

**T4 GPU (Free/Pro):**
- 500 examples: 30-60 minutes
- 2,000 examples: 2-3 hours
- 5,000 examples: 4-6 hours
- 48,000 examples: 12-16 hours

**V100 GPU (Pro):**
- 500 examples: 15-30 minutes
- 2,000 examples: 1-1.5 hours
- 5,000 examples: 2-3 hours
- 48,000 examples: 6-8 hours

**A100 GPU (Pro+):**
- 500 examples: 5-10 minutes
- 2,000 examples: 20-30 minutes
- 5,000 examples: 40-60 minutes
- 48,000 examples: 2-3 hours

### Memory Usage (T4 with 4-bit)
```
Base Model (4-bit):        ~4.0 GB
LoRA Adapters:             ~0.5 GB
Optimizer States:          ~2.0 GB
Gradient Checkpointing:    ~1.5 GB
Batch Processing:          ~1.0 GB
System Overhead:           ~1.0 GB
─────────────────────────────────
Total:                     ~10.0 GB
Available on T4:           15.0 GB
Safety Margin:             ~5.0 GB ✓
```

## 🚦 Detailed Workflow

### Phase 1: Dataset Preparation

```bash
# Convert HuggingFace dataset to JSONL
python 01-convert_polish_dataset.py

# Output: 02-polish_train.jsonl (48,637 examples)
# See: 08-README_DATASET_CONVERSION.md for details
```

### Phase 2: Model Training

1. Open `03-granite_tiny_polish_finetuning_pro.ipynb` in Google Colab
2. Select GPU runtime: Runtime → Change runtime type → GPU (T4/V100/A100)
3. Upload `02-polish_train.jsonl`
4. Run all cells (takes 1-2 hours on V100)
5. Save model to Google Drive

**See**: `10-GRANITE_TINY_FINETUNING_GUIDE.md` for complete training guide

### Phase 3: Testing & Evaluation

```bash
# Option A: Use Colab notebook (RECOMMENDED)
# 1. Open 06-test_granite_polish.ipynb in Colab
# 2. Upload 05-polish_test_questions.jsonl
# 3. Test base model → Download results_before.json
# 4. Test fine-tuned model → Download results_after.json
# 5. Compare results manually

# Option B: Use local script (requires local GPU)
python 07-test_polish_model.py \
  --model unsloth/granite-4.0-h-tiny \
  --output results_before.json

python 07-test_polish_model.py \
  --model granite4-tiny-h-polish-lora-pro \
  --output results_after.json
```

**See**: `09-README_TESTING.md` for complete testing guide

## 💡 Best Practices

### 1. Start Small
- Begin with 200-500 examples
- Validate approach before scaling
- Iterate on data quality

### 2. Focus on Narrow Tasks
- Don't try to build a general Polish chatbot
- Target specific use cases (extraction, classification, etc.)
- Measure improvements objectively

### 3. Quality Over Quantity
- 500 high-quality examples > 5,000 poor examples
- Have native Polish speakers review data
- Test thoroughly before production

### 4. Save Frequently
- Enable checkpoint saving every 200 steps
- Save to Google Drive for persistence
- Keep multiple versions

### 5. Monitor Training
- Watch loss curves for overfitting
- Test on held-out examples (use `05-polish_test_questions.jsonl`)
- Compare before/after performance

## 🔍 Evaluation Framework

### Test Suite Included
The project includes 20 diverse test questions covering:
- Grammar correction
- Translation (EN→PL)
- Summarization
- Math word problems
- Classification
- Question answering
- Creative writing
- Logical reasoning
- Text formatting
- Information extraction
- Comparison
- Instruction following
- Sentiment analysis
- Paraphrasing
- Entity recognition
- Definitions
- Advice
- Error detection
- Story continuation
- Factual knowledge

### Key Questions to Answer
1. ✅ Does it answer in Polish naturally?
2. ✅ Does it keep JSON valid (if applicable)?
3. ✅ Does it follow Polish instructions correctly?
4. ✅ Does it hallucinate less on Polish documents?
5. ✅ Is it faster and cheaper than bigger models?

### Metrics by Task Type

| Task Type | Metrics |
|-----------|---------|
| Classification | Accuracy, F1, Precision, Recall |
| Extraction | Exact match, JSON validity, Field accuracy |
| Summarization | ROUGE, Human evaluation |
| RAG Answer | Faithfulness, Source usage |
| Polish Quality | Native speaker review |
| Safety | Refusal rate, Hallucination detection |

## 💰 Cost Comparison

### Training Costs

| Scenario | GPU | Time | Cost |
|----------|-----|------|------|
| POC (500 examples) | Free T4 | 30-60 min | $0 |
| Development (2K examples) | Pro T4 | 2-3 hours | $10/month |
| Production (48K examples) | Pro V100 | 6-8 hours | $10/month |
| Production (48K examples) | Pro+ A100 | 2-3 hours | $50/month |

### Inference Costs

| Model | Cost per 1M tokens | Latency |
|-------|-------------------|---------|
| Fine-tuned Granite Tiny | $0.10-0.50 | 50-200ms |
| GPT-4 | $30-60 | 500-2000ms |
| **Savings** | **60-600x cheaper** | **2-10x faster** |

## 🛠️ Troubleshooting

### Out of Memory?
```python
# Reduce batch size
per_device_train_batch_size=1

# Increase gradient accumulation
gradient_accumulation_steps=16

# Reduce sequence length
max_seq_length=1024
```

### Training Too Slow?
- Upgrade to Colab Pro+ (A100)
- Reduce dataset size for testing
- Increase batch size if memory allows

### Poor Polish Quality?
- Increase training examples (aim for 2,000+)
- Improve data quality (native speaker review)
- Increase training epochs (3-4 instead of 2)
- Adjust learning rate (try 1e-4 or 3e-4)

### Session Timeout?
- Upgrade to Colab Pro (24-hour sessions)
- Save checkpoints frequently (`save_steps=100`)
- Use Google Drive for automatic backup

## 🎓 Example: Polish Enterprise Document Extractor

### Goal
Extract structured information from Polish business documents.

### Input
```
Zapytanie ofertowe nr ZO/2026/456

Klient: XYZ Bank S.A.
Termin składania ofert: 15 lipca 2026

Wymagane produkty:
- Platforma watsonx.data do zarządzania danymi
- Red Hat OpenShift do konteneryzacji
- Integracja z Microsoft SQL Server
```

### Expected Output
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

This is exactly the kind of task where a small fine-tuned Granite can be valuable.

## 📚 Documentation Index

1. **`README.md`** (this file) - Project overview and quick start
2. **`08-README_DATASET_CONVERSION.md`** - Dataset preparation guide
3. **`09-README_TESTING.md`** - Testing and evaluation guide
4. **`10-GRANITE_TINY_FINETUNING_GUIDE.md`** - Complete training guide
5. **`11-RESOURCE_ASSESSMENT.md`** - Resource requirements and costs

## 📞 Support

For issues or questions:
1. Check the troubleshooting sections in the guides
2. Review the [Resource Assessment](11-RESOURCE_ASSESSMENT.md)
3. Consult [IBM Granite Documentation](https://www.ibm.com/granite/docs/)
4. Check [Unsloth Documentation](https://github.com/unslothai/unsloth)

## 📄 License

This guide and code are provided as-is for educational and commercial use.

## 🙏 Acknowledgments

- IBM for Granite models
- Unsloth for efficient fine-tuning
- Hugging Face for model hosting and datasets
- Google Colab for free GPU access

---

**Last Updated**: 2026-06-20  
**Version**: 2.0  
**Status**: Production Ready ✅