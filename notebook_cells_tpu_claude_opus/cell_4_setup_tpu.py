#!/usr/bin/env python3
"""
=============================================================================
  TPU CELL 4: INITIALIZE ACCELERATOR RUNTIME
=============================================================================
  - Initializes TPU/XLA (or falls back to GPU/CPU)
  - Configures environment variables for optimal performance
  - Assesses memory capacity
  - Reports device topology and readiness
=============================================================================
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path

for _p in [
    os.path.dirname(os.path.abspath(__file__)),
    os.path.join(os.getcwd(), "notebook_cells_tpu"),
    ".",
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tpu_config__1_ import MODEL_ID, DTYPE, CACHE_DIR, IS_COLAB, IS_KAGGLE, MAX_NEW_TOKENS

print("\n" + "=" * 60)
print("  TPU CELL 4: INITIALIZE ACCELERATOR RUNTIME")
print("=" * 60)

# ---------------------------------------------------------------------------
# 4.1  Set XLA Environment Variables (BEFORE importing torch_xla)
# ---------------------------------------------------------------------------
print("\n  Setting environment variables...")

os.environ["XLA_USE_BF16"] = "1" if DTYPE == "bfloat16" else "0"
os.environ["XLA_TENSOR_ALLOCATOR_MAXSIZE"] = "500000000"  # 500MB chunks
os.environ.setdefault("XLA_IR_DEBUG", "0")
os.environ.setdefault("XLA_HLO_DEBUG", "0")
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HOME"] = CACHE_DIR
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

print(f"  XLA_USE_BF16:   {os.environ.get('XLA_USE_BF16')}")
print(f"  CACHE_DIR:      {CACHE_DIR}")

# ---------------------------------------------------------------------------
# 4.2  Initialize Device (TPU > GPU > CPU)
# ---------------------------------------------------------------------------
print("\n  Initializing device...")

import torch

device = None
device_type = "cpu"
device_info = {}

# Try TPU
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    # Verify it works
    _test = torch.tensor([1.0], device=device)
    _ = _test.cpu()
    device_type = "tpu"
    print(f"  TPU initialized: {device}")
    print(f"  torch:           {torch.__version__}")
    print(f"  torch_xla:       {torch_xla.__version__}")
except ImportError:
    print("  torch_xla not available.")
except Exception as e:
    print(f"  TPU init failed: {e}")

# Try GPU
if device_type == "cpu" and torch.cuda.is_available():
    device = torch.device("cuda")
    device_type = "gpu"
    print(f"  GPU initialized: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM:            {torch.cuda.get_device_properties(0).total_mem / (1024**3):.1f} GB")

# CPU fallback
if device_type == "cpu":
    device = torch.device("cpu")
    print(f"  CPU mode (no accelerator detected)")
    print(f"  torch:           {torch.__version__}")

# ---------------------------------------------------------------------------
# 4.3  Memory Assessment
# ---------------------------------------------------------------------------
print("\n  Memory Assessment...")

# Estimate model size
model_lower = MODEL_ID.lower()
model_sizes = {
    "72b": (144, "72B"), "32b": (64, "32B"), "14b": (28, "14B"),
    "7b": (14, "7B"), "3b": (6, "3B"), "1.5b": (3, "1.5B"),
    "0.5b": (1, "0.5B"),
}
model_bf16_gb, model_params = 14, "?B"  # default
for tag, (size, params) in model_sizes.items():
    if tag in model_lower:
        model_bf16_gb, model_params = size, params
        break

if device_type == "tpu":
    if IS_KAGGLE:
        tpu_ver, hbm_per_core, cores = "v3-8", 16, 8
    elif IS_COLAB:
        tpu_ver, hbm_per_core, cores = "v2-8", 8, 8
    else:
        tpu_ver, hbm_per_core, cores = "unknown", 8, 8
    total_mem_gb = cores * hbm_per_core
    usable_gb = int(total_mem_gb * 0.85)
    fits = model_bf16_gb <= usable_gb
    print(f"  TPU:            {tpu_ver} ({cores} cores x {hbm_per_core} GB = {total_mem_gb} GB)")
    device_info = {"tpu_version": tpu_ver, "cores": cores, "total_hbm_gb": total_mem_gb}

elif device_type == "gpu":
    total_mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    usable_gb = int(total_mem_gb * 0.90)
    fits = model_bf16_gb <= usable_gb
    print(f"  GPU:            {torch.cuda.get_device_name(0)} ({total_mem_gb:.1f} GB)")
    device_info = {"gpu": torch.cuda.get_device_name(0), "vram_gb": round(total_mem_gb, 1)}

else:
    import psutil
    total_mem_gb = psutil.virtual_memory().total / (1024**3) if hasattr(psutil, 'virtual_memory') else 12
    usable_gb = int(total_mem_gb * 0.70)
    fits = model_bf16_gb <= usable_gb
    print(f"  CPU:            {total_mem_gb:.1f} GB RAM")
    device_info = {"ram_gb": round(total_mem_gb, 1)}

print(f"  Model:          {MODEL_ID} ({model_params} params)")
print(f"  Model size:     ~{model_bf16_gb} GB in {DTYPE}")
print(f"  Usable memory:  ~{usable_gb} GB")
print(f"  Fits in memory: {'YES' if fits else 'NO - consider a smaller model'}")

if not fits:
    print(f"\n  WARNING: {MODEL_ID} ({model_bf16_gb} GB) may not fit in {usable_gb} GB.")
    print(f"  Recommendations:")
    print(f"    - Use Qwen/Qwen2.5-7B-Instruct (14 GB)")
    if device_type != "tpu":
        print(f"    - Switch to TPU runtime: Runtime > Change runtime type > TPU")

# ---------------------------------------------------------------------------
# 4.4  Disk Space Check
# ---------------------------------------------------------------------------
try:
    total, used, free = shutil.disk_usage(CACHE_DIR)
    print(f"\n  Disk: {free/(1024**3):.1f} GB free / {total/(1024**3):.1f} GB total")
    if free / (1024**3) < model_bf16_gb * 1.2:
        print(f"  WARNING: May need ~{int(model_bf16_gb*1.2)} GB disk for model weights.")
except Exception:
    pass

# ---------------------------------------------------------------------------
# 4.5  Summary
# ---------------------------------------------------------------------------
config_summary = {
    "device_type": device_type,
    "device": str(device),
    "model": MODEL_ID,
    "dtype": DTYPE,
    "model_size_gb": model_bf16_gb,
    "fits_in_memory": fits,
    **device_info,
}

print("\n  Configuration:")
print(json.dumps(config_summary, indent=4))

print("\n  TPU CELL 4 COMPLETE")
print("=" * 60)
