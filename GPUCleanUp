# ============================================================================
# GPU CLEANUP - RUN THIS BEFORE EACH TEST
# ============================================================================
"""
Run this cell BEFORE running test_llama.py or test_pixtral.py
This clears GPU memory from previous runs.
"""

import torch
import gc

print("🧹 Cleaning GPU memory...")

# Clear Python garbage
gc.collect()

# Clear CUDA cache
torch.cuda.empty_cache()

# Reset peak memory stats
torch.cuda.reset_peak_memory_stats()

# Show memory status
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free = total - allocated

    print(f"✅ GPU cleaned!")
    print(f"   Total VRAM: {total:.1f} GB")
    print(f"   Free: {free:.1f} GB")
    print(f"   Allocated: {allocated:.1f} GB")
    print(f"   Reserved: {reserved:.1f} GB")
else:
    print("❌ No GPU available")
