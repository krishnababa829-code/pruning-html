#!/usr/bin/env python3
"""
=============================================================================
  CELL 1: CLONE REPOSITORY & DETECT ENVIRONMENT
=============================================================================
  - Detects Colab vs Kaggle vs Local
  - Clones the GitLab repo (or pulls if already cloned)
  - Verifies GPU availability and VRAM
  - Sets working directory
=============================================================================
"""

import os
import sys
import subprocess
import platform
import json
import time

print("\n" + "=" * 60)
print("  CELL 1: CLONE REPO & DETECT ENVIRONMENT")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1.1  Detect Environment
# ---------------------------------------------------------------------------
env_info = {
    "platform": platform.system(),
    "python": platform.python_version(),
    "is_colab": os.path.exists("/content") and not os.path.exists("/kaggle"),
    "is_kaggle": os.path.exists("/kaggle/working"),
    "gpu_available": False,
    "gpu_name": "N/A",
    "gpu_vram_gb": 0.0,
    "cuda_version": "N/A",
}

# Check GPU
try:
    import torch
    if torch.cuda.is_available():
        env_info["gpu_available"] = True
        env_info["gpu_name"] = torch.cuda.get_device_name(0)
        env_info["gpu_vram_gb"] = round(
            torch.cuda.get_device_properties(0).total_mem / (1024**3), 1
        )
        env_info["cuda_version"] = torch.version.cuda or "N/A"
except ImportError:
    pass

# Also check via nvidia-smi
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode == 0:
        env_info["nvidia_smi"] = result.stdout.strip()
except (FileNotFoundError, subprocess.TimeoutExpired):
    env_info["nvidia_smi"] = "nvidia-smi not found"

print("\n  Environment Detection:")
for k, v in env_info.items():
    print(f"    {k:20s}: {v}")

if not env_info["gpu_available"]:
    print("\n  WARNING: No GPU detected!")
    if env_info["is_colab"]:
        print("  Go to: Runtime > Change runtime type > T4 GPU")
    elif env_info["is_kaggle"]:
        print("  Go to: Settings > Accelerator > GPU T4 x2")
    print("  Continuing without GPU (will be VERY slow)...")
else:
    print(f"\n  GPU Ready: {env_info['gpu_name']} ({env_info['gpu_vram_gb']} GB VRAM)")

# ---------------------------------------------------------------------------
# 1.2  Clone Repository
# ---------------------------------------------------------------------------
REPO_URL = "https://gitlab.com/cruzesolutions-group/cruzesolutions-project.git"

if env_info["is_colab"]:
    WORK_DIR = "/content/cruzesolutions-project"
elif env_info["is_kaggle"]:
    WORK_DIR = "/kaggle/working/cruzesolutions-project"
else:
    WORK_DIR = os.path.abspath("cruzesolutions-project")

print(f"\n  Cloning to: {WORK_DIR}")

if os.path.exists(os.path.join(WORK_DIR, ".git")):
    print("  Repo already exists. Pulling latest...")
    subprocess.run(["git", "-C", WORK_DIR, "pull", "--ff-only"], check=False)
else:
    subprocess.run(["git", "clone", REPO_URL, WORK_DIR], check=True)
    print("  Clone complete.")

# Change to repo directory
os.chdir(WORK_DIR)
print(f"  Working directory: {os.getcwd()}")

# List files
print("\n  Repository files:")
for f in sorted(os.listdir(".")):
    if not f.startswith("."):
        size = os.path.getsize(f) if os.path.isfile(f) else 0
        print(f"    {f:40s} {size:>8,d} bytes" if size else f"    {f:40s} [dir]")

# Add to Python path so notebook_cells/ imports work from ANY cell
# This is critical for Colab: each cell shares the same kernel,
# so paths added here persist for all subsequent cells.
for p in [
    WORK_DIR,
    os.path.join(WORK_DIR, "notebook_cells"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

print(f"  Python path updated ({len(sys.path)} entries)")

# Verify the imports will work for subsequent cells
try:
    import notebook_config
    print(f"  notebook_config: importable (env={notebook_config.IS_COLAB and 'Colab' or notebook_config.IS_KAGGLE and 'Kaggle' or 'Local'})")
except ImportError:
    print("  WARNING: notebook_config not importable - check repo structure")

try:
    import timing_tracker
    print("  timing_tracker:  importable")
except ImportError:
    print("  WARNING: timing_tracker not importable - check repo structure")

print("\n  CELL 1 COMPLETE")
print("=" * 60)
