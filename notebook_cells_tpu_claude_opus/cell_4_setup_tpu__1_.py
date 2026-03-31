#!/usr/bin/env python3
"""
=============================================================================
  TPU CELL 4: INITIALIZE ACCELERATOR RUNTIME
=============================================================================
  - Initializes TPU/XLA (or falls back to GPU/CPU)
  - Configures XLA environment variables for optimal performance
  - Reports ACCURATE per-core memory (not misleading total)
  - Determines sharding strategy: single-core vs FSDP multi-core
  - Assesses whether the model fits and how

  CRITICAL INSIGHT:
    TPU v2-8 = 8 cores x 8 GB = 64 GB total
    BUT xm.xla_device() -> xla:0 = ONE core = 8 GB
    A 14B model (28 GB) CANNOT fit on a single 8 GB core.
    You MUST shard across cores using FSDP or use xmp.spawn().
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

# XLA_USE_BF16: auto-cast float32 ops to bfloat16 (TPU-native)
os.environ["XLA_USE_BF16"] = "1" if DTYPE == "bfloat16" else "0"

# Larger allocation chunks reduce fragmentation on TPU HBM
os.environ["XLA_TENSOR_ALLOCATOR_MAXSIZE"] = "500000000"  # 500MB

# Disable XLA debug logging (reduces overhead)
os.environ.setdefault("XLA_IR_DEBUG", "0")
os.environ.setdefault("XLA_HLO_DEBUG", "0")

# PJRT runtime (required for modern torch_xla multi-core)
os.environ.setdefault("PJRT_DEVICE", "TPU")

# HuggingFace cache
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HOME"] = CACHE_DIR
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

print(f"  XLA_USE_BF16:   {os.environ.get('XLA_USE_BF16')}")
print(f"  PJRT_DEVICE:    {os.environ.get('PJRT_DEVICE')}")
print(f"  CACHE_DIR:      {CACHE_DIR}")

# ---------------------------------------------------------------------------
# 4.2  Initialize Device (TPU > GPU > CPU)
# ---------------------------------------------------------------------------
print("\n  Initializing device...")

import torch

device = None
device_type = "cpu"
device_info = {}
tpu_num_cores = 1
tpu_hbm_per_core_gb = 0

# --- Try TPU ---
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    # Verify it works
    _test = torch.tensor([1.0], device=device)
    _ = _test.cpu()
    del _test
    device_type = "tpu"

    # Detect core count
    try:
        import torch_xla.runtime as xr
        tpu_num_cores = xr.global_device_count()
    except Exception:
        tpu_num_cores = xm.xrt_world_size() if hasattr(xm, "xrt_world_size") else 8

    # Determine TPU version and per-core HBM
    if IS_KAGGLE:
        tpu_ver = "v3-8"
        tpu_hbm_per_core_gb = 16
    elif IS_COLAB:
        tpu_ver = "v2-8"
        tpu_hbm_per_core_gb = 8
    else:
        tpu_ver = "unknown"
        tpu_hbm_per_core_gb = 8

    total_hbm_gb = tpu_num_cores * tpu_hbm_per_core_gb

    print(f"  TPU initialized: {device}")
    print(f"  torch:           {torch.__version__}")
    print(f"  torch_xla:       {torch_xla.__version__}")
    print(f"  TPU version:     {tpu_ver}")
    print(f"  Total cores:     {tpu_num_cores}")

except ImportError:
    print("  torch_xla not available.")
except Exception as e:
    print(f"  TPU init failed: {e}")

# --- Try GPU ---
if device_type == "cpu" and torch.cuda.is_available():
    device = torch.device("cuda")
    device_type = "gpu"
    print(f"  GPU initialized: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM:            {torch.cuda.get_device_properties(0).total_mem / (1024**3):.1f} GB")

# --- CPU fallback ---
if device_type == "cpu":
    device = torch.device("cpu")
    print(f"  CPU mode (no accelerator detected)")
    print(f"  torch:           {torch.__version__}")

# ---------------------------------------------------------------------------
# 4.3  Memory Assessment (ACCURATE per-core analysis)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  MEMORY ASSESSMENT (CRITICAL)")
print("=" * 60)

# Estimate model size in bfloat16
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
    # ===== THIS IS THE KEY INSIGHT =====
    # xm.xla_device() returns xla:0 = ONE core.
    # model.to(xla:0) loads the ENTIRE model onto that ONE core.
    # A TPU v2-8 has 8 cores x 8 GB = 64 GB total,
    # but each core only has 8 GB of HBM.
    #
    # To use all 64 GB, you MUST shard the model across cores
    # using FSDP (Fully Sharded Data Parallel).

    usable_per_core_gb = tpu_hbm_per_core_gb * 0.80  # 80% usable after XLA overhead
    fits_single_core = model_bf16_gb <= usable_per_core_gb

    # With FSDP, each core holds 1/N of the model + activation memory
    sharded_per_core_gb = model_bf16_gb / tpu_num_cores
    fits_sharded = sharded_per_core_gb <= usable_per_core_gb

    print(f"\n  TPU:              {tpu_ver}")
    print(f"  Cores:            {tpu_num_cores}")
    print(f"  HBM per core:     {tpu_hbm_per_core_gb} GB")
    print(f"  Usable per core:  ~{usable_per_core_gb:.1f} GB (after XLA overhead)")
    print(f"  Total HBM:        {total_hbm_gb} GB (but NOT usable as one pool!)")
    print(f"")
    print(f"  Model:            {MODEL_ID} ({model_params})")
    print(f"  Model size:       ~{model_bf16_gb} GB in {DTYPE}")
    print(f"")
    print(f"  --- Single Core (model.to(xla:0)) ---")
    print(f"  Fits?             {'YES' if fits_single_core else 'NO'}")
    print(f"  Reason:           {model_bf16_gb} GB model vs {tpu_hbm_per_core_gb} GB core")
    print(f"")
    print(f"  --- FSDP Sharded (across {tpu_num_cores} cores) ---")
    print(f"  Per-core shard:   ~{sharded_per_core_gb:.1f} GB")
    print(f"  Fits?             {'YES' if fits_sharded else 'NO'}")

    if not fits_single_core and fits_sharded:
        print(f"")
        print(f"  STRATEGY: FSDP sharding required.")
        print(f"  Cell 5 will use XlaFullyShardedDataParallel to distribute")
        print(f"  the model across all {tpu_num_cores} TPU cores.")
    elif not fits_single_core and not fits_sharded:
        print(f"")
        print(f"  WARNING: Model too large even with FSDP!")
        print(f"  {sharded_per_core_gb:.1f} GB/core > {usable_per_core_gb:.1f} GB usable/core")
        print(f"  Recommendations:")
        print(f"    - Use Qwen/Qwen2.5-7B-Instruct (14 GB, ~1.75 GB/core with FSDP)")
        print(f"    - Use Qwen/Qwen2.5-3B (6 GB, fits on single core)")
    else:
        print(f"")
        print(f"  STRATEGY: Single-core loading (model fits in one core).")

    # Store strategy for Cell 5
    use_fsdp = not fits_single_core and fits_sharded
    device_info = {
        "tpu_version": tpu_ver, "cores": tpu_num_cores,
        "hbm_per_core_gb": tpu_hbm_per_core_gb,
        "total_hbm_gb": total_hbm_gb,
        "fits_single_core": fits_single_core,
        "fits_sharded": fits_sharded,
        "use_fsdp": use_fsdp,
    }

elif device_type == "gpu":
    total_mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    usable_gb = total_mem_gb * 0.90
    fits = model_bf16_gb <= usable_gb
    use_fsdp = False
    print(f"  GPU:              {torch.cuda.get_device_name(0)} ({total_mem_gb:.1f} GB)")
    print(f"  Model size:       ~{model_bf16_gb} GB in {DTYPE}")
    print(f"  Fits?             {'YES' if fits else 'NO'}")
    device_info = {"gpu": torch.cuda.get_device_name(0), "vram_gb": round(total_mem_gb, 1)}

else:
    use_fsdp = False
    try:
        import psutil
        total_mem_gb = psutil.virtual_memory().total / (1024**3)
    except Exception:
        total_mem_gb = 12
    usable_gb = total_mem_gb * 0.70
    fits = model_bf16_gb <= usable_gb
    print(f"  CPU:              {total_mem_gb:.1f} GB RAM")
    print(f"  Model size:       ~{model_bf16_gb} GB in {DTYPE}")
    print(f"  Fits?             {'YES' if fits else 'NO'}")
    device_info = {"ram_gb": round(total_mem_gb, 1)}

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
    "use_fsdp": use_fsdp,
    **device_info,
}

print("\n  Configuration:")
print(json.dumps(config_summary, indent=4))

print("\n  TPU CELL 4 COMPLETE")
print("=" * 60)
