# Fix for Notebook 06 Kernel Crash

## Problem
The kernel crashes with "kernel restarted" errors when running [`06-test_granite_polish.ipynb`](06-test_granite_polish.ipynb:1). This is caused by **Out of Memory (OOM)** issues during model loading or inference.

## Root Causes

1. **Insufficient GPU memory** - The 4GB Granite model with 4-bit quantization still requires significant GPU memory
2. **No memory cleanup** - GPU cache not cleared between operations
3. **No error handling** - Crashes instead of graceful failure
4. **Memory accumulation** - Processing 20 questions without clearing intermediate results

## Solutions

### Immediate Fix: Add Memory Management Cell

**Add this cell BEFORE Step 6 (Load Model):**

```python
# Clear GPU memory before loading model
import torch
import gc

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print(f"✓ GPU cache cleared")
    print(f"GPU Memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### Fix 1: Improve Model Loading (Replace Step 6 cell)

```python
## Step 6: Load Model
import torch
import gc
from unsloth import FastLanguageModel

# Clear GPU cache before loading
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print(f"GPU Memory before loading: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

print(f"\nLoading model: {MODEL_NAME}")
print("This may take 2-3 minutes...\n")

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    print("✓ Model loaded successfully!")
    print(f"Model size: {model.get_memory_footprint() / 1e9:.2f} GB")
    
    if torch.cuda.is_available():
        print(f"GPU Memory after loading: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("\n⚠ ERROR: Out of GPU memory!")
        print("\nSolutions:")
        print("1. Restart runtime: Runtime → Restart runtime")
        print("2. Use a GPU with more memory (T4 has 16GB, V100 has 16GB, A100 has 40GB)")
        print("3. Close other notebooks using GPU")
        print("4. Try reducing max_seq_length to 1024")
        raise
    else:
        raise
```

### Fix 2: Add Memory Cleanup to generate_response (Update Step 7)

```python
def generate_response(prompt, max_new_tokens=512, temperature=0.7):
    """Generate response from the model with memory management."""
    import torch
    import gc
    
    try:
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Generate
        with torch.no_grad():  # Disable gradient computation
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
        
        # Clean up
        del inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()
        
        # Extract assistant response
        assistant_start = response.find("<|start_of_role|>assistant<|end_of_role|>")
        if assistant_start != -1:
            assistant_response = response[assistant_start + len("<|start_of_role|>assistant<|end_of_role|>"):].strip()
            assistant_response = assistant_response.replace("<|end_of_text|>", "").strip()
            return assistant_response
        
        return response
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # Clear cache and try again with reduced tokens
            torch.cuda.empty_cache()
            gc.collect()
            print(f"⚠ OOM error, retrying with reduced tokens...")
            return generate_response(prompt, max_new_tokens=max_new_tokens//2, temperature=temperature)
        raise
```

### Fix 3: Add Batch Processing with Memory Checks (Update Step 8)

```python
## Step 8: Run Tests with Memory Management
from datetime import datetime
import torch
import gc

print(f"\n{'='*60}")
print(f"TESTING: {MODEL_NAME}")
print(f"{'='*60}")
print(f"Total questions: {len(questions)}")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

results = []
BATCH_SIZE = 5  # Process in batches to manage memory

for batch_start in range(0, len(questions), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(questions))
    batch_questions = questions[batch_start:batch_end]
    
    print(f"\n{'='*60}")
    print(f"Processing batch {batch_start//BATCH_SIZE + 1} (questions {batch_start+1}-{batch_end})")
    print(f"{'='*60}")
    
    for i, question in enumerate(batch_questions, batch_start + 1):
        print(f"\n[{i}/{len(questions)}] Testing...")
        result = test_single_question(question, show_details=True)
        results.append(result)
        
        # Memory check every 5 questions
        if i % 5 == 0:
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated(0) / 1e9
                memory_reserved = torch.cuda.memory_reserved(0) / 1e9
                print(f"\n📊 GPU Memory: {memory_used:.2f}GB used, {memory_reserved:.2f}GB reserved")
    
    # Clear cache after each batch
    torch.cuda.empty_cache()
    gc.collect()
    print(f"\n✓ Batch {batch_start//BATCH_SIZE + 1} complete, memory cleared")

print(f"\n{'='*60}")
print("✓ Testing complete!")
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")
```

## Alternative: Reduce Memory Requirements

If crashes persist, modify Step 6 to use smaller settings:

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=1024,  # Reduced from 2048
    dtype=None,
    load_in_4bit=True,
)
```

And in generate_response:

```python
max_new_tokens=256,  # Reduced from 512
```

## Preventive Measures

1. **Before running the notebook:**
   - Restart runtime: `Runtime → Restart runtime`
   - Verify GPU: Run `!nvidia-smi` to check available memory
   - Close other notebooks using GPU

2. **During execution:**
   - Monitor GPU memory in each cell output
   - If memory usage > 90%, restart runtime before continuing

3. **GPU Requirements:**
   - **Minimum:** T4 GPU (16GB) - should work with fixes
   - **Recommended:** V100 (16GB) or A100 (40GB) - more headroom
   - **Free Colab:** May need to reduce max_seq_length to 1024

## Testing the Fix

1. Restart runtime completely
2. Run cells 1-5 normally
3. Add the memory management cell before Step 6
4. Run Step 6 with improved error handling
5. Run Step 7 with updated generate_response
6. Run Step 8 with batch processing
7. Monitor GPU memory in outputs

## If Still Crashing

1. **Reduce sequence length to 1024**
2. **Reduce max_new_tokens to 256**
3. **Process 3 questions at a time instead of 5**
4. **Use Colab Pro for better GPU access**
5. **Test with fewer questions first (e.g., first 5)**

## Quick Test Script

Add this cell after Step 6 to verify model works:

```python
# Quick test to verify model loaded correctly
test_prompt = format_prompt("Przetłumacz na polski.", "Hello, world!")
test_response = generate_response(test_prompt, max_new_tokens=50)
print(f"Test response: {test_response}")
print("✓ Model is working!")
```

---

**Note:** The kernel restart warnings about frozen modules and debugger validation are normal and can be ignored. The actual crash is the "kernel restarted" message, which indicates OOM.