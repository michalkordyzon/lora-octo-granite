# Step 2 "Crash" is INTENTIONAL - Not a Bug

## What You're Seeing

The kernel restart at **Step 2** with these warnings:
```
WARNING - Debugger warning: It seems that frozen modules are being used...
WARNING - kernel restarted
INFO - AsyncIOLoopKernelRestarter: restarting kernel (1/5)
```

**This is NORMAL and EXPECTED behavior!**

## Why It Happens

Line 94 in Step 2 intentionally kills the kernel:
```python
import os
os.kill(os.getpid(), 9)
```

This forces a runtime restart to ensure newly installed packages (unsloth, transformers, etc.) load correctly with their dependencies.

## What You Should Do

1. **Wait for the restart to complete** (10-20 seconds)
2. **Skip Step 2** after restart (it's already done)
3. **Continue from Step 3** onwards

## The Warnings Are Harmless

- `frozen modules` warning - Python debugger notice, can be ignored
- `kernel restarted` - Expected, this is what we wanted
- `AsyncIOLoopKernelRestarter` - Colab's automatic restart mechanism

## If You Want to Avoid the Restart

Replace Step 2 cell with this version (no auto-restart):

```python
# Install Unsloth and let it handle PyTorch dependencies
!pip install --upgrade --no-cache-dir unsloth
!pip install "transformers<=5.5.0" accelerate bitsandbytes

print("✓ Installation complete!")
print("\n⚠ IMPORTANT: Please manually restart runtime:")
print("   Runtime → Restart runtime")
print("\nAfter restart, continue from Step 3.")

# Don't auto-restart - let user do it manually
# import os
# os.kill(os.getpid(), 9)
```

Then manually restart: `Runtime → Restart runtime`

## Real Crashes vs. Expected Restart

**Expected Restart (Step 2):**
- Happens immediately after `os.kill(os.getpid(), 9)`
- Shows "kernel restarted" message
- Runtime comes back online in 10-20 seconds
- You can continue from Step 3

**Real Crash (OOM during model loading):**
- Happens during Step 6 (Load Model) or Step 8 (Run Tests)
- Shows "kernel restarted" but keeps restarting repeatedly
- May show "Your session crashed" message
- Cannot continue - need to apply memory fixes

## Summary

✅ **Step 2 restart = NORMAL** - Just continue from Step 3
❌ **Step 6/8 restart = PROBLEM** - Apply memory management fixes from [`NOTEBOOK_CRASH_FIX.md`](NOTEBOOK_CRASH_FIX.md:1)

The crash you reported at Step 2 is actually the notebook working as designed. The real issue would be if it crashes again later during model loading or testing.