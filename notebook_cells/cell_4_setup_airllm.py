#!/usr/bin/env python3
"""
=============================================================================
  CELL 4: SETUP AirLLM FOR COLAB / KAGGLE
=============================================================================
  - Configures AirLLM for the detected environment
  - Sets up cache directories
  - Configures memory optimization for free T4 GPU
  - Verifies AirLLM can initialize
=============================================================================
"""

import os
import sys
import gc
import time
import json
from pathlib import Path

try:
    from notebook_config import (
        MODEL_ID, COMPRESSION, CACHE_DIR,
        IS_COLAB, IS_KAGGLE, MAX_NEW_TOKENS,
    )
except ImportError:
    sys.path.insert(0, os.path.join(os.getcwd(), "notebook_cells"))
    from notebook_config import (
        MODEL_ID, COMPRESSION, CACHE_DIR,
        IS_COLAB, IS_KAGGLE, MAX_NEW_TOKENS,
    )

print("\n" + "=" * 60)
print("  CELL 4: SETUP AirLLM")
print("=" * 60)

# ---------------------------------------------------------------------------
# 4.1  Environment-Specific Configuration
# ---------------------------------------------------------------------------
print("\n  Configuring for environment...")

if IS_COLAB:
    print("  Platform: Google Colab")
    print("  Free T4 GPU: 15 GB VRAM, ~80 GB disk")
    # Colab-specific optimizations
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
    os.environ["HF_HOME"] = CACHE_DIR

elif IS_KAGGLE:
    print("  Platform: Kaggle")
    print("  Free T4 GPU: 16 GB VRAM, ~20 GB scratch")
    os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
    os.environ["HF_HOME"] = CACHE_DIR

else:
    print("  Platform: Local")

# Create cache directory
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
print(f"  Cache directory: {CACHE_DIR}")

# Check available disk space
try:
    import shutil
    total, used, free = shutil.disk_usage(CACHE_DIR)
    print(f"  Disk space: {free / (1024**3):.1f} GB free / {total / (1024**3):.1f} GB total")

    # Warn if low disk space (72B model needs ~40-50 GB for 4-bit)
    if free / (1024**3) < 30:
        print("  WARNING: Low disk space! 72B model needs ~40-50 GB for 4-bit weights.")
        print("  Consider using a smaller model (Qwen2.5-7B or 32B).")
except Exception:
    pass

# ---------------------------------------------------------------------------
# 4.2  GPU Memory Optimization
# ---------------------------------------------------------------------------
print("\n  GPU Memory Optimization...")

try:
    import torch
    if torch.cuda.is_available():
        # Clear any existing allocations
        gc.collect()
        torch.cuda.empty_cache()

        free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
        total_mem = torch.cuda.mem_get_info()[1] / (1024**3)
        print(f"  VRAM: {free_mem:.1f} GB free / {total_mem:.1f} GB total")

        # Set memory fraction to leave some headroom
        torch.cuda.set_per_process_memory_fraction(0.95)
        print("  Memory fraction set to 95%")

        # Enable TF32 for faster computation on T4
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("  TF32 enabled for faster computation")
    else:
        print("  No GPU - skipping GPU optimization")
except ImportError:
    print("  PyTorch not available - skipping GPU optimization")

# ---------------------------------------------------------------------------
# 4.3  Verify AirLLM Installation
# ---------------------------------------------------------------------------
print("\n  Verifying AirLLM...")

try:
    import airllm
    print(f"  AirLLM version: {airllm.__version__}")

    from airllm import AutoModel as AirAutoModel
    print("  AirLLM AutoModel: importable")

    print(f"\n  Model to load: {MODEL_ID}")
    print(f"  Compression:   {COMPRESSION}")
    print(f"  Max new tokens: {MAX_NEW_TOKENS}")

    # Estimate memory requirements
    model_lower = MODEL_ID.lower()
    if "72b" in model_lower:
        if COMPRESSION == "4bit":
            est_disk = "~40 GB disk, ~2 GB peak VRAM per layer"
        elif COMPRESSION == "8bit":
            est_disk = "~75 GB disk, ~3 GB peak VRAM per layer"
        else:
            est_disk = "~140 GB disk, ~5 GB peak VRAM per layer"
    elif "32b" in model_lower:
        est_disk = "~18 GB disk (4bit), ~1 GB peak VRAM per layer"
    elif "7b" in model_lower:
        est_disk = "~4 GB disk (4bit), ~0.5 GB peak VRAM per layer"
    else:
        est_disk = "varies by model size"

    print(f"  Estimated resources: {est_disk}")

except ImportError as e:
    print(f"  ERROR: AirLLM not installed: {e}")
    print("  Run: !pip install airllm")
    raise

# ---------------------------------------------------------------------------
# 4.4  Configuration Summary
# ---------------------------------------------------------------------------
config_summary = {
    "model_id": MODEL_ID,
    "compression": COMPRESSION,
    "cache_dir": CACHE_DIR,
    "max_new_tokens": MAX_NEW_TOKENS,
    "platform": "Colab" if IS_COLAB else "Kaggle" if IS_KAGGLE else "Local",
}

print("\n  AirLLM Configuration:")
print(json.dumps(config_summary, indent=4))

print("\n  CELL 4 COMPLETE - AirLLM is ready")
print("=" * 60)
