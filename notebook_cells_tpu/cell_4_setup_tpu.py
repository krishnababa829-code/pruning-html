#!/usr/bin/env python3
"""
=============================================================================
  TPU CELL 4: INITIALIZE TPU / XLA RUNTIME
=============================================================================
  - Initializes PyTorch/XLA
  - Configures XLA environment variables for optimal performance
  - Sets up TPU device mesh for model sharding
  - Configures memory management for large models
  - Reports TPU topology and readiness
=============================================================================
"""

import os
import sys
import time
import json
from pathlib import Path

for _p in ["notebook_cells_tpu", os.path.join(os.getcwd(), "notebook_cells_tpu"), "."]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tpu_config import MODEL_ID, DTYPE, CACHE_DIR, IS_COLAB, IS_KAGGLE, MAX_NEW_TOKENS

print("\n" + "=" * 60)
print("  TPU CELL 4: INITIALIZE XLA RUNTIME")
print("=" * 60)

# ---------------------------------------------------------------------------
# 4.1  XLA Environment Variables (must be set BEFORE importing torch_xla)
# ---------------------------------------------------------------------------
print("\n  Setting XLA environment variables...")

# Optimize XLA compilation
os.environ["XLA_USE_BF16"] = "1" if DTYPE == "bfloat16" else "0"
os.environ["XLA_TENSOR_ALLOCATOR_MAXSIZE"] = "100000000"  # 100MB allocation chunks

# Reduce XLA compilation overhead
os.environ.setdefault("XLA_IR_DEBUG", "0")
os.environ.setdefault("XLA_HLO_DEBUG", "0")

# HuggingFace cache
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HOME"] = CACHE_DIR
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

print(f"  XLA_USE_BF16:   {os.environ.get('XLA_USE_BF16')}")
print(f"  CACHE_DIR:      {CACHE_DIR}")

# ---------------------------------------------------------------------------
# 4.2  Initialize XLA Device
# ---------------------------------------------------------------------------
print("\n  Initializing XLA...")

import torch
import torch_xla
import torch_xla.core.xla_model as xm

tpu_device = xm.xla_device()
print(f"  XLA device:     {tpu_device}")
print(f"  torch:          {torch.__version__}")
print(f"  torch_xla:      {torch_xla.__version__}")

# ---------------------------------------------------------------------------
# 4.3  TPU Memory Assessment
# ---------------------------------------------------------------------------
print("\n  TPU Memory Assessment...")

if IS_KAGGLE:
    tpu_ver = "v3-8"
    hbm_per_core = 16
    cores = 8
elif IS_COLAB:
    tpu_ver = "v2-8"
    hbm_per_core = 8
    cores = 8
else:
    tpu_ver = "unknown"
    hbm_per_core = 8
    cores = 8

total_hbm = cores * hbm_per_core

# Estimate model memory needs
model_lower = MODEL_ID.lower()
if "72b" in model_lower:
    model_bf16_gb = 144
    model_params = "72B"
elif "32b" in model_lower:
    model_bf16_gb = 64
    model_params = "32B"
elif "14b" in model_lower:
    model_bf16_gb = 28
    model_params = "14B"
elif "7b" in model_lower:
    model_bf16_gb = 14
    model_params = "7B"
else:
    model_bf16_gb = 64
    model_params = "?B"

fits = model_bf16_gb <= total_hbm * 0.85  # 85% usable after overhead

print(f"  TPU:            {tpu_ver} ({cores} cores x {hbm_per_core} GB = {total_hbm} GB)")
print(f"  Model:          {MODEL_ID} ({model_params} params)")
print(f"  Model size:     ~{model_bf16_gb} GB in {DTYPE}")
print(f"  Available HBM:  ~{int(total_hbm * 0.85)} GB (85% usable)")
print(f"  Fits in memory: {'YES' if fits else 'NO'}")

if not fits:
    print(f"\n  WARNING: {MODEL_ID} ({model_bf16_gb} GB) may not fit in {total_hbm} GB HBM.")
    print(f"  Recommendations:")
    if total_hbm <= 64:
        print(f"    - Use Qwen/Qwen2.5-32B-Instruct (64 GB) or smaller")
        print(f"    - Or switch to Kaggle TPU v3-8 (128 GB)")
    print(f"    - Or use notebook_cells/ (GPU version) with AirLLM")

# ---------------------------------------------------------------------------
# 4.4  Disk Space Check
# ---------------------------------------------------------------------------
try:
    import shutil
    total, used, free = shutil.disk_usage(CACHE_DIR)
    print(f"\n  Disk: {free/(1024**3):.1f} GB free / {total/(1024**3):.1f} GB total")
    if free / (1024**3) < model_bf16_gb * 1.2:
        print(f"  WARNING: May need ~{int(model_bf16_gb*1.2)} GB disk for model weights.")
except Exception:
    pass

# ---------------------------------------------------------------------------
# 4.5  Summary
# ---------------------------------------------------------------------------
tpu_config_summary = {
    "tpu_version": tpu_ver,
    "cores": cores,
    "total_hbm_gb": total_hbm,
    "model": MODEL_ID,
    "dtype": DTYPE,
    "model_size_gb": model_bf16_gb,
    "fits_in_memory": fits,
    "backend": "PyTorch/XLA",
}

print("\n  Configuration:")
print(json.dumps(tpu_config_summary, indent=4))

print("\n  TPU CELL 4 COMPLETE")
print("=" * 60)
