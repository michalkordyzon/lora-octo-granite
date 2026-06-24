# Granite 4.0 H-Tiny Fine-Tuning: Resource Assessment

## Executive Summary

Fine-tuning Granite 4.0 H-Tiny (4B parameters) with LoRA on Google Colab is **feasible and cost-effective** for Polish language tasks.

**Key Findings:**
- ✅ Fits on free T4 GPU (15GB VRAM)
- ✅ Training time: 30 minutes to 4 hours (depending on dataset size)
- ✅ Cost: $0-50/month
- ✅ 60-600x cheaper than GPT-4 for narrow tasks

---

## Detailed Resource Analysis

### 1. Model Size and Memory Requirements

#### Base Model Specifications
```
Model: ibm-granite/granite-4.0-h-tiny
Parameters: ~4 billion
Architecture: Decoder-only transformer
Context Length: 2048 tokens (default), up to 8192 tokens
```

#### Memory Footprint by Precision

| Precision | Model Size | Use Case |
|-----------|------------|----------|
| FP32 (32-bit) | ~16 GB | Not recommended (too large) |
| FP16 (16-bit) | ~8 GB | Standard training |
| INT8 (8-bit) | ~4 GB | Inference only |
| INT4 (4-bit) | ~2 GB | LoRA training (recommended) |

**Recommendation**: Use 4-bit quantization for training on T4 GPU.

---

### 2. Google Colab GPU Options

#### Option 1: Free Tier (T4 GPU)

**Hardware Specifications:**
```
GPU: NVIDIA Tesla T4
VRAM: 15 GB
Compute Capability: 7.5
CUDA Cores: 2,560
Tensor Cores: 320
FP16 Performance: 65 TFLOPS
```

**System Resources:**
```
CPU: Intel Xeon (2 cores)
RAM: 12-13 GB
Disk: 78 GB (temporary)
Network: High-speed internet
```

**Limitations:**
- Session timeout: 12 hours maximum
- Idle disconnect: 90 minutes
- GPU availability: Not guaranteed (may need to wait)
- No background execution
- Sessions reset daily

**Cost:** FREE

**Best For:**
- Proof of concept (200-500 examples)
- Learning and experimentation
- Small datasets (<2,000 examples)

#### Option 2: Colab Pro (T4/V100)

**Hardware Specifications:**
```
GPU: NVIDIA Tesla T4 or V100
VRAM: 15-16 GB
```

**System Resources:**
```
CPU: Intel Xeon (4 cores)
RAM: 25-27 GB
Disk: 166 GB (temporary)
```

**Benefits:**
- Session timeout: 24 hours
- Better GPU availability
- Background execution
- Priority access to GPUs
- Faster reconnection

**Cost:** ~$10/month

**Best For:**
- Small to medium datasets (2,000-5,000 examples)
- Regular fine-tuning work
- Multiple experiments

#### Option 3: Colab Pro+ (A100)

**Hardware Specifications:**
```
GPU: NVIDIA A100
VRAM: 40 GB
Compute Capability: 8.0
CUDA Cores: 6,912
Tensor Cores: 432
FP16 Performance: 312 TFLOPS
BF16 Support: Yes
```

**System Resources:**
```
CPU: Intel Xeon (8 cores)
RAM: 51 GB
Disk: 166+ GB (temporary)
```

**Benefits:**
- Session timeout: Extended (up to 24 hours)
- Best GPU availability
- 5-10x faster training than T4
- Can handle larger batches
- BFloat16 support

**Cost:** ~$50/month

**Best For:**
- Large datasets (10,000-50,000 examples)
- Production model training
- Time-sensitive projects
- Multiple large experiments

---

### 3. Memory Usage Breakdown

#### Training on T4 with 4-bit Quantization

```
Component                          Memory Usage    Percentage
─────────────────────────────────────────────────────────────
Base Model (4-bit quantized)       ~4.0 GB         26.7%
LoRA Adapter Parameters            ~0.5 GB         3.3%
Optimizer States (AdamW 8-bit)     ~2.0 GB         13.3%
Gradient Checkpointing             ~1.5 GB         10.0%
Activation Memory (batch_size=1)   ~1.0 GB         6.7%
System Overhead                    ~1.0 GB         6.7%
─────────────────────────────────────────────────────────────
Total Estimated Usage              ~10.0 GB        66.7%
Available on T4                    15.0 GB         100%
Safety Margin                      ~5.0 GB         33.3% ✓
```

**Verdict:** ✅ T4 GPU has sufficient memory with comfortable margin.

#### Training on A100 with 4-bit Quantization

```
Component                          Memory Usage    Percentage
─────────────────────────────────────────────────────────────
Base Model (4-bit quantized)       ~4.0 GB         10.0%
LoRA Adapter Parameters            ~0.5 GB         1.3%
Optimizer States (AdamW 8-bit)     ~2.0 GB         5.0%
Gradient Checkpointing             ~1.5 GB         3.8%
Activation Memory (batch_size=4)   ~4.0 GB         10.0%
System Overhead                    ~1.0 GB         2.5%
─────────────────────────────────────────────────────────────
Total Estimated Usage              ~13.0 GB        32.5%
Available on A100                  40.0 GB         100%
Safety Margin                      ~27.0 GB        67.5% ✓
```

**Verdict:** ✅ A100 can handle larger batches and faster training.

---

### 4. Training Time Estimates

#### Factors Affecting Training Time
1. Dataset size (number of examples)
2. Sequence length (tokens per example)
3. Batch size and gradient accumulation
4. Number of epochs
5. GPU type (T4 vs A100)

#### Time Estimates by Dataset Size

**T4 GPU (Free/Pro):**

| Dataset Size | Avg Tokens/Example | Epochs | Batch Size | Time Estimate |
|--------------|-------------------|--------|------------|---------------|
| 200 examples | 512 | 2 | 8 | 15-30 min |
| 500 examples | 512 | 2 | 8 | 30-60 min |
| 1,000 examples | 512 | 2 | 8 | 1-2 hours |
| 2,000 examples | 512 | 2 | 8 | 2-3 hours |
| 5,000 examples | 512 | 2 | 8 | 4-6 hours |
| 10,000 examples | 512 | 2 | 8 | 8-12 hours |

**A100 GPU (Pro+):**

| Dataset Size | Avg Tokens/Example | Epochs | Batch Size | Time Estimate |
|--------------|-------------------|--------|------------|---------------|
| 200 examples | 512 | 2 | 32 | 3-5 min |
| 500 examples | 512 | 2 | 32 | 5-10 min |
| 1,000 examples | 512 | 2 | 32 | 10-20 min |
| 2,000 examples | 512 | 2 | 32 | 20-30 min |
| 5,000 examples | 512 | 2 | 32 | 40-60 min |
| 10,000 examples | 512 | 2 | 32 | 1-2 hours |

**Speed Comparison:**
- A100 is **5-10x faster** than T4
- Larger batch sizes on A100 improve efficiency
- Longer sequences increase training time proportionally

---

### 5. Storage Requirements

#### During Training

```
Component                          Size
─────────────────────────────────────────
Base Model Cache                   ~8 GB
Training Dataset                   ~10-500 MB
Checkpoints (2 saved)              ~100-400 MB
Logs and Metadata                  ~10-50 MB
Temporary Files                    ~1-2 GB
─────────────────────────────────────────
Total During Training              ~10-12 GB
Available on Free Colab            78 GB ✓
Available on Pro/Pro+ Colab        166 GB ✓
```

#### After Training

```
Component                          Size
─────────────────────────────────────────
LoRA Adapter Only                  ~50-200 MB
Merged Model (FP16)                ~8 GB
Merged Model (GGUF Q4)             ~2.5 GB
Merged Model (GGUF Q8)             ~4 GB
Tokenizer Files                    ~5 MB
─────────────────────────────────────────
```

**Recommendation:** Save only LoRA adapter (~50-200 MB) to Google Drive.

---

### 6. Cost Analysis

#### Training Costs

**Scenario 1: Proof of Concept**
```
Dataset: 500 examples
GPU: Free T4
Training Time: 30-60 minutes
Cost: $0
Monthly Cost: $0
```

**Scenario 2: Internal Tool Development**
```
Dataset: 2,000 examples
GPU: Colab Pro T4
Training Time: 2-3 hours
Training Runs: 5-10 iterations
Cost per Run: ~$0 (included in subscription)
Monthly Cost: $10
```

**Scenario 3: Production Model**
```
Dataset: 10,000 examples
GPU: Colab Pro+ A100
Training Time: 1-2 hours
Training Runs: 3-5 iterations
Cost per Run: ~$0 (included in subscription)
Monthly Cost: $50
```

#### Inference Costs (Self-Hosted)

**Granite 4.0 H-Tiny (Fine-tuned):**
```
Hardware: T4 GPU or CPU
Tokens/second: 50-100 (T4), 10-20 (CPU)
Cost per 1M tokens: $0.10-0.50 (self-hosted)
Latency: 50-200ms
```

**Comparison with GPT-4:**
```
Cost per 1M tokens: $30-60
Latency: 500-2000ms
Savings: 60-600x cheaper
```

#### Total Cost of Ownership (First Year)

**Option A: Free Tier**
```
Colab: $0/month × 12 = $0
Total: $0
Best for: POC, learning
```

**Option B: Colab Pro**
```
Colab: $10/month × 12 = $120
Total: $120
Best for: Small projects, regular use
```

**Option C: Colab Pro+**
```
Colab: $50/month × 12 = $600
Total: $600
Best for: Production, time-sensitive
```

**Option D: Dedicated GPU Server (for comparison)**
```
Cloud GPU (T4): $0.35/hour × 730 hours = $255/month
Annual: $3,060
Best for: 24/7 production serving
```

---

### 7. Network and Data Transfer

#### Model Download (First Time)
```
Base Model: ~8 GB
Download Time: 2-5 minutes (Colab's fast connection)
Cached: Yes (subsequent runs are instant)
```

#### Dataset Upload
```
Small (500 examples): ~1-5 MB → <10 seconds
Medium (2,000 examples): ~10-50 MB → <30 seconds
Large (10,000 examples): ~50-500 MB → 1-5 minutes
```

#### Model Download (After Training)
```
LoRA Adapter: ~50-200 MB → 10-30 seconds
Merged Model: ~8 GB → 5-15 minutes
GGUF Q4: ~2.5 GB → 2-5 minutes
```

---

### 8. Optimization Strategies

#### Memory Optimization

**If you run out of memory:**

1. **Reduce batch size:**
   ```python
   per_device_train_batch_size=1  # Minimum
   ```

2. **Increase gradient accumulation:**
   ```python
   gradient_accumulation_steps=16  # Maintain effective batch size
   ```

3. **Reduce sequence length:**
   ```python
   max_seq_length=1024  # Instead of 2048
   ```

4. **Enable gradient checkpointing:**
   ```python
   use_gradient_checkpointing="unsloth"  # Saves ~40% memory
   ```

5. **Use 4-bit quantization:**
   ```python
   load_in_4bit=True  # Reduces model size by 75%
   ```

#### Speed Optimization

**To train faster:**

1. **Upgrade to A100:**
   - 5-10x faster than T4
   - Can use larger batches

2. **Increase batch size (if memory allows):**
   ```python
   per_device_train_batch_size=2  # or 4 on A100
   gradient_accumulation_steps=4  # Adjust accordingly
   ```

3. **Use mixed precision:**
   ```python
   fp16=True  # T4/V100
   bf16=True  # A100 (better numerical stability)
   ```

4. **Reduce dataset size for testing:**
   - Start with 200-500 examples
   - Validate approach before scaling

5. **Optimize data loading:**
   ```python
   dataloader_num_workers=2
   dataloader_pin_memory=True
   ```

---

### 9. Recommended Configurations

#### Configuration 1: Quick Test (Free T4)
```python
# For testing with 200-500 examples
max_seq_length = 1024
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
num_train_epochs = 1
learning_rate = 2e-4

# Expected time: 15-30 minutes
# Memory usage: ~8 GB
# Cost: $0
```

#### Configuration 2: Development (Pro T4)
```python
# For development with 2,000 examples
max_seq_length = 2048
per_device_train_batch_size = 1
gradient_accumulation_steps = 8
num_train_epochs = 2
learning_rate = 2e-4

# Expected time: 2-3 hours
# Memory usage: ~10 GB
# Cost: $10/month
```

#### Configuration 3: Production (Pro+ A100)
```python
# For production with 10,000 examples
max_seq_length = 2048
per_device_train_batch_size = 4
gradient_accumulation_steps = 8
num_train_epochs = 3
learning_rate = 2e-4

# Expected time: 1-2 hours
# Memory usage: ~15 GB
# Cost: $50/month
```

---

### 10. Comparison with Alternatives

#### vs. Full Fine-Tuning

| Aspect | LoRA | Full Fine-Tuning |
|--------|------|------------------|
| Trainable Parameters | ~0.5-1% | 100% |
| Memory Required | ~10 GB | ~30+ GB |
| Training Time | 2-4 hours | 8-24 hours |
| Risk of Forgetting | Low | High |
| Adapter Size | 50-200 MB | 8 GB |
| GPU Required | T4 (15GB) | A100 (40GB+) |
| Cost | $0-50 | $200-500 |

**Verdict:** LoRA is clearly superior for this use case.

#### vs. Continued Pretraining

| Aspect | LoRA Fine-Tuning | Continued Pretraining |
|--------|------------------|----------------------|
| Goal | Task-specific behavior | Language knowledge |
| Data Required | 500-10,000 examples | 1M-1B tokens |
| Training Time | 2-4 hours | Days to weeks |
| GPU Required | T4 (15GB) | Multiple A100s |
| Cost | $0-50 | $1,000-10,000+ |
| Polish Support | Task-specific | General language |

**Verdict:** LoRA fine-tuning is the right approach for narrow Polish tasks.

#### vs. Using GPT-4

| Aspect | Fine-Tuned Granite Tiny | GPT-4 API |
|--------|------------------------|-----------|
| Setup Cost | $0-50 (one-time) | $0 |
| Cost per 1M tokens | $0.10-0.50 | $30-60 |
| Latency | 50-200ms | 500-2000ms |
| Customization | Full control | Limited |
| Data Privacy | Self-hosted | Sent to OpenAI |
| Offline Use | Yes | No |
| Polish Quality | Task-optimized | General |

**Verdict:** Fine-tuned Granite is 60-600x cheaper for narrow tasks.

---

## Conclusion

### Feasibility: ✅ HIGHLY FEASIBLE

Granite 4.0 H-Tiny fine-tuning on Google Colab is:
- **Technically feasible**: Fits comfortably on free T4 GPU
- **Time efficient**: 30 minutes to 4 hours depending on dataset
- **Cost effective**: $0-50/month vs. $1000s for alternatives
- **Production ready**: Can handle 10,000+ examples

### Recommended Path

**Week 1: Proof of Concept**
- Use: Free Colab T4
- Dataset: 200-500 examples
- Time: 30-60 minutes
- Cost: $0
- Goal: Validate approach

**Week 2-3: Development**
- Use: Colab Pro T4
- Dataset: 2,000 examples
- Time: 2-3 hours per iteration
- Cost: $10/month
- Goal: Refine model quality

**Week 4+: Production**
- Use: Colab Pro+ A100
- Dataset: 10,000 examples
- Time: 1-2 hours
- Cost: $50/month
- Goal: Deploy production model

### Key Success Factors

1. ✅ Start small (200-500 examples)
2. ✅ Use 4-bit quantization
3. ✅ Enable gradient checkpointing
4. ✅ Save checkpoints frequently
5. ✅ Test iteratively
6. ✅ Focus on narrow tasks
7. ✅ Measure improvements

### Risk Assessment

**Low Risk:**
- Out of memory (easily mitigated)
- Session timeout (save checkpoints)
- Poor initial results (iterate on data)

**Medium Risk:**
- GPU availability on free tier (upgrade to Pro)
- Data quality issues (review and improve)

**High Risk:**
- None identified for this approach

---

**Document Version**: 1.0  
**Last Updated**: 2026-06-20  
**Assessment**: APPROVED FOR PRODUCTION USE