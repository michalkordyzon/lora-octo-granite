# Granite 4.0 H-Tiny Fine-Tuning for Polish Tasks

Complete guide and resources for fine-tuning IBM's Granite 4.0 H-Tiny model using LoRA on Google Colab for Polish language tasks.

## 🚀 Quick Start

1. **Open the Colab notebook**: [`granite_tiny_polish_finetuning.ipynb`](granite_tiny_polish_finetuning.ipynb)
2. **Upload your Polish training data** in JSONL format
3. **Run all cells** (takes 30 minutes to 4 hours depending on dataset size)
4. **Test your fine-tuned model** with Polish instructions

## 📋 What's Included

### 1. [Comprehensive Guide](GRANITE_TINY_FINETUNING_GUIDE.md)
- Complete step-by-step instructions
- Best practices and troubleshooting
- Evaluation framework
- Deployment options
- Example use cases

### 2. [Ready-to-Use Colab Notebook](granite_tiny_polish_finetuning.ipynb)
- Pre-configured for T4 GPU
- All code cells ready to run
- Interactive testing included
- Automatic model saving

### 3. [Resource Assessment](RESOURCE_ASSESSMENT.md)
- Detailed memory analysis
- Training time estimates
- Cost breakdown
- GPU comparison
- Optimization strategies

## ✅ Key Features

- **Free to Start**: Works on Google Colab free tier (T4 GPU)
- **Fast Training**: 30 minutes to 4 hours depending on dataset
- **Cost Effective**: $0-50/month vs. $1000s for alternatives
- **Production Ready**: Can handle 10,000+ examples
- **60-600x Cheaper**: Than GPT-4 for narrow Polish tasks

## 📊 Resource Requirements

### Minimum (Free Tier)
- **GPU**: T4 (15GB VRAM) - FREE
- **Dataset**: 200-500 examples
- **Time**: 30-60 minutes
- **Cost**: $0

### Recommended (Production)
- **GPU**: A100 (40GB VRAM) - $50/month
- **Dataset**: 10,000 examples
- **Time**: 1-2 hours
- **Cost**: $50/month

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

| Goal | Examples | Training Time (T4) | Cost |
|------|----------|-------------------|------|
| Proof of concept | 200-500 | 30-60 min | $0 |
| Useful internal model | 2,000-5,000 | 2-4 hours | $10/month |
| Strong narrow model | 10,000-50,000 | 8-20 hours | $50/month |

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
- ✅ **Fast Training**: 2-4 hours vs. days for full fine-tuning
- ✅ **Low Risk**: Minimal catastrophic forgetting
- ✅ **Small Adapters**: 50-200MB vs. 8GB full model
- ✅ **Cost Effective**: $0-50 vs. $1000s

## 📈 Expected Results

### Training Time by Dataset Size

**T4 GPU (Free/Pro):**
- 500 examples: 30-60 minutes
- 2,000 examples: 2-3 hours
- 5,000 examples: 4-6 hours
- 10,000 examples: 8-12 hours

**A100 GPU (Pro+):**
- 500 examples: 5-10 minutes
- 2,000 examples: 20-30 minutes
- 5,000 examples: 40-60 minutes
- 10,000 examples: 1-2 hours

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

## 🚦 Getting Started

### Step 1: Prepare Your Data

Create a JSONL file with your Polish examples:

```python
import json

data = [
    {
        "instruction": "Wyodrębnij dane z tekstu i zwróć JSON.",
        "input": "Faktura nr FV/123/2026 z dnia 12.06.2026 na kwotę 4500 PLN dla firmy ABC Sp. z o.o.",
        "output": '{"typ":"faktura","numer":"FV/123/2026","data":"2026-06-12","kwota":"4500 PLN","firma":"ABC Sp. z o.o."}'
    },
    # Add more examples...
]

with open('polish_train.jsonl', 'w', encoding='utf-8') as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
```

### Step 2: Open Colab Notebook

1. Open [`granite_tiny_polish_finetuning.ipynb`](granite_tiny_polish_finetuning.ipynb) in Google Colab
2. Select GPU runtime: Runtime → Change runtime type → GPU (T4)
3. Upload your `polish_train.jsonl` file

### Step 3: Run Training

Simply run all cells in the notebook. The process will:
1. Install dependencies (3-5 minutes)
2. Load model (2-3 minutes)
3. Prepare dataset (1 minute)
4. Train model (30 minutes to 4 hours)
5. Save adapter (1 minute)
6. Test model (interactive)

### Step 4: Evaluate and Deploy

Test your model with Polish instructions and deploy using one of these options:
- **Option A**: Load base model + adapter (good for testing)
- **Option B**: Merge and export (good for production)
- **Option C**: Convert to GGUF (good for local deployment)

## 📚 Documentation

### Main Documents
1. **[GRANITE_TINY_FINETUNING_GUIDE.md](GRANITE_TINY_FINETUNING_GUIDE.md)** - Complete guide with all details
2. **[RESOURCE_ASSESSMENT.md](RESOURCE_ASSESSMENT.md)** - Detailed resource analysis
3. **[granite_tiny_polish_finetuning.ipynb](granite_tiny_polish_finetuning.ipynb)** - Ready-to-use notebook

### Quick References
- [IBM Granite Documentation](https://www.ibm.com/granite/docs/)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Granite Model Card](https://huggingface.co/ibm-granite/granite-4.0-h-tiny)

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
- Test on held-out examples
- Compare before/after performance

## 🔍 Evaluation

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
| Production (10K examples) | Pro+ A100 | 1-2 hours | $50/month |

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

## 📞 Support

For issues or questions:
1. Check the [Troubleshooting section](GRANITE_TINY_FINETUNING_GUIDE.md#troubleshooting)
2. Review the [Resource Assessment](RESOURCE_ASSESSMENT.md)
3. Consult [IBM Granite Documentation](https://www.ibm.com/granite/docs/)

## 📄 License

This guide and code are provided as-is for educational and commercial use.

## 🙏 Acknowledgments

- IBM for Granite models
- Unsloth for efficient fine-tuning
- Hugging Face for model hosting
- Google Colab for free GPU access

---

**Last Updated**: 2026-06-20  
**Version**: 1.0  
**Status**: Production Ready ✅