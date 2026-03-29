#!/usr/bin/env python3
"""
=============================================================================
  TPU CELL 1: CLONE REPOSITORY & DETECT TPU
=============================================================================
  - Detects TPU type (v2-8 / v3-8) and core count
  - Clones the GitLab repo
  - Validates TPU is accessible via XLA
  - Reports HBM (High Bandwidth Memory) per core
=============================================================================
"""

import os
import sys
import subprocess
import platform
import time

print("\n" + "=" * 60)
print("  TPU CELL 1: CLONE REPO & DETECT TPU")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1.1  Detect Environment
# ---------------------------------------------------------------------------
is_colab = os.path.exists("/content") and not os.path.exists("/kaggle")
is_kaggle = os.path.exists("/kaggle/working")

print(f"\n  Platform:  {'Colab' if is_colab else 'Kaggle' if is_kaggle else 'Local'}")
print(f"  Python:    {platform.python_version()}")
print(f"  OS:        {platform.system()} {platform.release()}")

# ---------------------------------------------------------------------------
# 1.2  Detect TPU
# ---------------------------------------------------------------------------
print("\n  Detecting TPU...")

tpu_available = False
tpu_type = "unknown"
tpu_cores = 0
tpu_hbm_per_core_gb = 0

# Check for TPU environment variables (set by Colab/Kaggle)
tpu_name = os.environ.get("TPU_NAME", "")
tpu_cluster = os.environ.get("TPU_CLUSTER_NAME", "")
colab_tpu = os.environ.get("COLAB_TPU_ADDR", "")

if tpu_name or colab_tpu:
    print(f"  TPU_NAME:       {tpu_name or 'N/A'}")
    print(f"  COLAB_TPU_ADDR: {colab_tpu or 'N/A'}")

# Try importing torch_xla to confirm TPU access
try:
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm

    # Get TPU device
    device = xm.xla_device()
    tpu_available = True

    # Detect core count
    tpu_cores = xm.xrt_world_size() if hasattr(xm, 'xrt_world_size') else 8

    # Determine TPU version from environment or device string
    device_str = str(device)
    if is_kaggle:
        tpu_type = "TPU v3-8"
        tpu_hbm_per_core_gb = 16
    elif is_colab:
        tpu_type = "TPU v2-8"
        tpu_hbm_per_core_gb = 8
    else:
        tpu_type = "TPU (unknown version)"
        tpu_hbm_per_core_gb = 8

    total_hbm_gb = tpu_cores * tpu_hbm_per_core_gb

    # Quick tensor test on TPU
    test_tensor = torch.randn(2, 2, device=device)
    result = (test_tensor @ test_tensor.T).cpu()

    print(f"\n  TPU DETECTED")
    print(f"  Type:           {tpu_type}")
    print(f"  Cores:          {tpu_cores}")
    print(f"  HBM/core:       {tpu_hbm_per_core_gb} GB")
    print(f"  Total HBM:      {total_hbm_gb} GB")
    print(f"  XLA device:     {device}")
    print(f"  Tensor test:    PASSED")
    print(f"  torch_xla:      {torch_xla.__version__}")

    # Model size guidance
    print(f"\n  Model Capacity Estimate ({total_hbm_gb} GB HBM):")
    if total_hbm_gb >= 128:
        print("    Qwen2.5-72B (bfloat16): YES (needs ~144 GB, fits with KV-cache mgmt)")
        print("    Qwen2.5-32B (bfloat16): YES (needs ~64 GB)")
    elif total_hbm_gb >= 64:
        print("    Qwen2.5-72B (bfloat16): NO  (needs ~144 GB, you have 64 GB)")
        print("    Qwen2.5-32B (bfloat16): YES (needs ~64 GB, tight fit)")
        print("    Qwen2.5-7B  (bfloat16): YES (needs ~14 GB)")
    else:
        print("    Qwen2.5-7B  (bfloat16): YES")
        print("    Larger models may not fit.")

except ImportError:
    print("\n  ERROR: torch_xla not installed.")
    print("  Ensure you selected TPU runtime:")
    if is_colab:
        print("    Runtime > Change runtime type > TPU")
    elif is_kaggle:
        print("    Settings > Accelerator > TPU v3-8")
    print("\n  Falling back: will attempt to install in Cell 2.")

except Exception as e:
    print(f"\n  TPU detection failed: {e}")
    print("  Ensure TPU runtime is selected.")

# Also check if GPU is available (user might have wrong runtime)
try:
    import torch
    if torch.cuda.is_available() and not tpu_available:
        print(f"\n  NOTE: GPU detected ({torch.cuda.get_device_name(0)}) but no TPU.")
        print("  You are in GPU mode. Use notebook_cells/ (GPU version) instead,")
        print("  or switch to TPU runtime.")
except Exception:
    pass

# ---------------------------------------------------------------------------
# 1.3  Clone Repository
# ---------------------------------------------------------------------------
REPO_URL = "https://gitlab.com/cruzesolutions-group/cruzesolutions-project.git"

if is_colab:
    WORK_DIR = "/content/cruzesolutions-project"
elif is_kaggle:
    WORK_DIR = "/kaggle/working/cruzesolutions-project"
else:
    WORK_DIR = os.path.abspath("cruzesolutions-project")

print(f"\n  Cloning to: {WORK_DIR}")

if os.path.exists(os.path.join(WORK_DIR, ".git")):
    print("  Repo exists. Pulling latest...")
    subprocess.run(["git", "-C", WORK_DIR, "pull", "--ff-only"], check=False)
else:
    subprocess.run(["git", "clone", REPO_URL, WORK_DIR], check=True)
    print("  Clone complete.")

os.chdir(WORK_DIR)

# Setup Python path
for p in [WORK_DIR, os.path.join(WORK_DIR, "notebook_cells_tpu")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Verify imports
try:
    import tpu_config
    print(f"  tpu_config:     importable")
except ImportError:
    print("  WARNING: tpu_config not found in path")

try:
    import timing_tracker
    print(f"  timing_tracker: importable")
except ImportError:
    print("  WARNING: timing_tracker not found")

print(f"  Working dir:    {os.getcwd()}")
print("\n  TPU CELL 1 COMPLETE")
print("=" * 60)
