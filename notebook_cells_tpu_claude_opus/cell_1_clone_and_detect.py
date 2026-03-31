#!/usr/bin/env python3
"""
=============================================================================
  TPU CELL 1: CLONE REPOSITORY & DETECT ACCELERATOR
=============================================================================
  - Detects environment (Colab / Kaggle / Local)
  - Detects accelerator (TPU via XLA, GPU via CUDA, or CPU)
  - Clones the target repository (no double-nesting)
  - Sets up Python path for all subsequent cells
  - Initializes timing tracker
=============================================================================
"""

import os
import sys
import subprocess
import platform
import time
import re
from urllib.parse import urlparse

print("\n" + "=" * 60)
print("  TPU CELL 1: CLONE REPO & DETECT ACCELERATOR")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1.1  Detect Environment
# ---------------------------------------------------------------------------
is_colab = os.path.exists("/content") and not os.path.exists("/kaggle")
is_kaggle = os.path.exists("/kaggle/working")
env_name = "Colab" if is_colab else "Kaggle" if is_kaggle else "Local"

print(f"\n  Platform:  {env_name}")
print(f"  Python:    {platform.python_version()}")
print(f"  OS:        {platform.system()} {platform.release()}")

# ---------------------------------------------------------------------------
# 1.2  Detect Accelerator (TPU > GPU > CPU)
# ---------------------------------------------------------------------------
print("\n  Detecting accelerator...")

accel_type = "cpu"  # default fallback
accel_device = None
accel_info = {}

# --- Try TPU first ---
tpu_name = os.environ.get("TPU_NAME", "")
colab_tpu = os.environ.get("COLAB_TPU_ADDR", "")

try:
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm

    accel_device = xm.xla_device()
    accel_type = "tpu"

    # Quick sanity: 2x2 matmul on TPU
    _t = torch.randn(2, 2, device=accel_device)
    _ = (_t @ _t.T).cpu()

    cores = xm.xrt_world_size() if hasattr(xm, "xrt_world_size") else 8
    if is_kaggle:
        tpu_ver, hbm_per_core = "v3-8", 16
    elif is_colab:
        tpu_ver, hbm_per_core = "v2-8", 8
    else:
        tpu_ver, hbm_per_core = "unknown", 8

    accel_info = {
        "type": "tpu", "version": tpu_ver, "cores": cores,
        "hbm_per_core_gb": hbm_per_core, "total_hbm_gb": cores * hbm_per_core,
        "torch_xla": torch_xla.__version__,
    }
    print(f"  TPU DETECTED: {tpu_ver} | {cores} cores | {cores * hbm_per_core} GB HBM")
    print(f"  torch_xla:    {torch_xla.__version__}")
    print(f"  XLA device:   {accel_device}")
    print(f"  Tensor test:  PASSED")

except ImportError:
    print("  torch_xla not available. Checking for GPU...")
except Exception as e:
    print(f"  TPU detection failed: {e}")
    print("  Checking for GPU...")

# --- Try GPU if no TPU ---
if accel_type == "cpu":
    try:
        import torch
        if torch.cuda.is_available():
            accel_type = "gpu"
            accel_device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            accel_info = {
                "type": "gpu", "name": gpu_name,
                "vram_gb": round(gpu_mem, 1),
                "torch": torch.__version__,
            }
            print(f"  GPU DETECTED: {gpu_name} | {gpu_mem:.1f} GB VRAM")
        else:
            accel_type = "cpu"
            accel_device = torch.device("cpu")
            accel_info = {"type": "cpu", "torch": torch.__version__}
            print(f"  CPU MODE (no TPU or GPU detected)")
    except ImportError:
        print("  PyTorch not installed. Will install in Cell 2.")
        accel_info = {"type": "cpu"}

print(f"\n  Accelerator:  {accel_type.upper()}")

# ---------------------------------------------------------------------------
# 1.3  Clone / Update Repository
# ---------------------------------------------------------------------------
# Import config for repo settings
# We need to add the script's own directory to sys.path first
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from tpu_config__1_ import REPO_URL, REPO_NAME, REPO_SLUG, WORK_DIR

print(f"\n  Repository:   {REPO_SLUG}")
print(f"  Clone URL:    {REPO_URL}")
print(f"  Work dir:     {WORK_DIR}")

if os.path.exists(os.path.join(WORK_DIR, ".git")):
    print("  Repo exists. Pulling latest...")
    subprocess.run(["git", "-C", WORK_DIR, "fetch", "origin"], check=False)
    pull = subprocess.run(
        ["git", "-C", WORK_DIR, "pull", "--ff-only"],
        capture_output=True, text=True, check=False,
    )
    if pull.returncode == 0:
        print("  Pull: OK")
    else:
        print(f"  Pull --ff-only failed (diverged?). Resetting to origin/main...")
        subprocess.run(
            ["git", "-C", WORK_DIR, "reset", "--hard", "origin/main"],
            check=False,
        )
else:
    print("  Cloning repository...")
    result = subprocess.run(
        ["git", "clone", REPO_URL, WORK_DIR],
        capture_output=True, text=True, check=False,
    )
    if result.returncode == 0:
        print("  Clone complete.")
    else:
        print(f"  Clone failed: {result.stderr[:200]}")
        # If WORK_DIR already exists as a non-git directory, that's OK
        if os.path.isdir(WORK_DIR):
            print(f"  Directory exists at {WORK_DIR}, continuing...")
        else:
            raise RuntimeError(f"Cannot clone repo and {WORK_DIR} does not exist.")

os.chdir(WORK_DIR)

# ---------------------------------------------------------------------------
# 1.4  Setup Python Path
# ---------------------------------------------------------------------------
for p in [
    WORK_DIR,
    os.path.join(WORK_DIR, "notebook_cells_tpu"),
    os.path.join(WORK_DIR, "air_llm"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# 1.5  Initialize Timing Tracker
# ---------------------------------------------------------------------------
try:
    from timing_tracker import configure_tracker
    tracker = configure_tracker(
        tracker_name="tpu_pipeline",
        repo_slug=REPO_SLUG,
        platform_name=env_name.lower(),
        device_name=accel_type,
        persist=True,
        artifacts_dir=os.path.join(WORK_DIR, "artifacts", "timing"),
        auto_print=True,
    )
    print(f"\n  timing_tracker: configured (run_id={tracker.run_id})")
except Exception as e:
    print(f"\n  WARNING: timing_tracker configure failed: {e}")

# ---------------------------------------------------------------------------
# 1.6  Verify Imports
# ---------------------------------------------------------------------------
for mod_name in ["tpu_config__1_", "timing_tracker"]:
    try:
        __import__(mod_name)
        print(f"  {mod_name}: importable")
    except ImportError as e:
        print(f"  WARNING: {mod_name} not importable: {e}")

print(f"\n  Working dir:  {os.getcwd()}")
print(f"  Accelerator:  {accel_type.upper()}")
print("\n  TPU CELL 1 COMPLETE")
print("=" * 60)
