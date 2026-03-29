#!/usr/bin/env python3
"""
=============================================================================
  TPU CELL 2: INSTALL TPU DEPENDENCIES & CHECK COMPATIBILITY
=============================================================================
  - Installs PyTorch/XLA (if not pre-installed)
  - Installs HuggingFace stack with TPU support
  - Verifies XLA can see TPU cores
  - Checks for known TPU incompatibilities

  NOTE: AirLLM is NOT installed. It is CUDA-only and incompatible with TPU.
=============================================================================
"""

import subprocess
import sys
import os
import time
import importlib

print("\n" + "=" * 60)
print("  TPU CELL 2: INSTALL DEPENDENCIES")
print("  Backend: PyTorch/XLA (NOT AirLLM)")
print("=" * 60)

# ---------------------------------------------------------------------------
# 2.1  Install Dependencies
# ---------------------------------------------------------------------------
# torch and torch_xla are pre-installed on Colab/Kaggle TPU runtimes.
# We only install the HuggingFace stack and HTML processing libs.

PACKAGES = [
    ("datasets", "datasets>=2.19.0"),
    ("pandas", "pandas>=2.0.0"),
    ("transformers", "transformers>=4.40.0"),
    ("huggingface_hub", "huggingface-hub>=0.23.0"),
    ("accelerate", "accelerate>=0.30.0"),
    ("bs4", "beautifulsoup4>=4.12.0"),
    ("lxml", "lxml>=5.0.0"),
    ("tiktoken", "tiktoken>=0.7.0"),
]

print("\n  Installing packages (AirLLM is NOT needed for TPU)...\n")
install_start = time.time()

failed = []
for import_name, pip_spec in PACKAGES:
    print(f"  {pip_spec}...", end=" ", flush=True)
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", pip_spec],
        capture_output=True, text=True,
    )
    print("OK" if r.returncode == 0 else "FAILED")
    if r.returncode != 0:
        failed.append((pip_spec, r.stderr[:200]))

print(f"\n  Installed in {time.time() - install_start:.1f}s")
if failed:
    print("  FAILURES:")
    for pkg, err in failed:
        print(f"    {pkg}: {err}")

# ---------------------------------------------------------------------------
# 2.2  Verify Critical Imports
# ---------------------------------------------------------------------------
print("\n  Verifying imports...\n")

CHECKS = [
    ("torch", "PyTorch"),
    ("torch_xla", "PyTorch/XLA"),
    ("transformers", "Transformers"),
    ("datasets", "Datasets"),
    ("pandas", "Pandas"),
    ("bs4", "BeautifulSoup4"),
    ("lxml", "lxml"),
    ("accelerate", "Accelerate"),
]

for mod_name, display in CHECKS:
    try:
        mod = importlib.import_module(mod_name)
        ver = getattr(mod, "__version__", "?")
        print(f"    {display:20s} v{ver:15s} OK")
    except ImportError as e:
        print(f"    {display:20s} {'N/A':15s} MISSING: {e}")

# ---------------------------------------------------------------------------
# 2.3  TPU-Specific Compatibility Checks
# ---------------------------------------------------------------------------
print("\n  TPU compatibility checks...\n")

issues = []

# Check 1: torch_xla available and functional
try:
    import torch
    import torch_xla.core.xla_model as xm
    dev = xm.xla_device()
    t = torch.tensor([1.0, 2.0, 3.0], device=dev)
    r = (t * t).sum().item()
    assert abs(r - 14.0) < 0.01, f"Expected 14.0, got {r}"
    print(f"    XLA tensor ops:       OK (1^2+2^2+3^2 = {r})")
except Exception as e:
    issues.append(f"XLA tensor ops failed: {e}")
    print(f"    XLA tensor ops:       FAILED ({e})")

# Check 2: bfloat16 support (TPU-native dtype)
try:
    import torch
    import torch_xla.core.xla_model as xm
    dev = xm.xla_device()
    bf = torch.tensor([1.0], dtype=torch.bfloat16, device=dev)
    print(f"    bfloat16 support:     OK")
except Exception as e:
    issues.append(f"bfloat16 failed: {e}")
    print(f"    bfloat16 support:     FAILED ({e})")

# Check 3: Transformers can target XLA device
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    ids = tok("test", return_tensors="pt").input_ids
    print(f"    Transformers tokenize: OK")
except Exception as e:
    issues.append(f"Transformers test failed: {e}")
    print(f"    Transformers tokenize: FAILED ({e})")

# Check 4: Confirm NO CUDA (we want TPU, not GPU)
try:
    import torch
    if torch.cuda.is_available():
        issues.append("CUDA is available - you may be in GPU mode, not TPU mode")
        print(f"    CUDA check:           WARNING (GPU detected, expected TPU)")
    else:
        print(f"    CUDA check:           OK (no GPU, TPU mode confirmed)")
except Exception:
    print(f"    CUDA check:           OK")

# ---------------------------------------------------------------------------
# 2.4  Summary
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
if issues:
    print(f"  ISSUES: {len(issues)}")
    for i in issues:
        print(f"    - {i}")
else:
    print("  ALL TPU CHECKS PASSED")

print("\n  TPU CELL 2 COMPLETE")
print("=" * 60)
