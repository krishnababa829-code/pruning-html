#!/usr/bin/env python3
"""
=============================================================================
  TPU CELL 2: INSTALL DEPENDENCIES & VERIFY ACCELERATOR
=============================================================================
  - Installs correct torch + torch_xla version pair for the runtime
  - Installs HuggingFace stack
  - Installs HTML processing libraries
  - Verifies all imports
  - Runs accelerator compatibility checks

  NOTE: AirLLM is NOT installed. It is CUDA-only.
=============================================================================
"""

import subprocess
import sys
import os
import time
import importlib

print("\n" + "=" * 60)
print("  TPU CELL 2: INSTALL DEPENDENCIES")
print("=" * 60)

# ---------------------------------------------------------------------------
# 2.1  Detect runtime to choose correct torch_xla version
# ---------------------------------------------------------------------------
is_colab = os.path.exists("/content") and not os.path.exists("/kaggle")
is_kaggle = os.path.exists("/kaggle/working")

# Check if torch_xla is already functional
torch_xla_ok = False
try:
    import torch
    import torch_xla.core.xla_model as xm
    dev = xm.xla_device()
    _ = torch.tensor([1.0], device=dev)
    torch_xla_ok = True
    print(f"\n  torch_xla already functional (torch={torch.__version__})")
except Exception as e:
    print(f"\n  torch_xla not functional: {type(e).__name__}")
    print("  Will attempt to install compatible versions...")

# ---------------------------------------------------------------------------
# 2.2  Install torch + torch_xla if needed (version-matched pair)
# ---------------------------------------------------------------------------
if not torch_xla_ok:
    print("\n  Installing PyTorch + XLA (version-matched)...")
    t0 = time.time()

    # On Colab TPU runtime, torch and torch_xla should be pre-installed.
    # If they're broken (ABI mismatch), we reinstall the matching pair.
    # The Colab TPU VM uses specific versions; we install the nightly
    # that matches the pre-installed torch version.
    try:
        import torch
        torch_ver = torch.__version__.split("+")[0]  # e.g. "2.10.0"
        print(f"  Existing torch: {torch.__version__}")
    except ImportError:
        torch_ver = None
        print("  No torch installed.")

    if is_colab:
        # Colab TPU: install torch_xla matching the pre-installed torch
        # Use the cloud-tpu-client approach for Colab
        cmds = [
            [sys.executable, "-m", "pip", "install", "-q",
             "torch_xla[tpu]", "-f", "https://storage.googleapis.com/libtpu-releases/index.html"],
        ]
    elif is_kaggle:
        # Kaggle TPU: torch_xla is pre-installed and usually works
        cmds = [
            [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "torch_xla"],
        ]
    else:
        # Local/other: install CPU torch + torch_xla for testing
        cmds = [
            [sys.executable, "-m", "pip", "install", "-q",
             "torch", "torch_xla"],
        ]

    for cmd in cmds:
        print(f"  Running: {' '.join(cmd[-3:])}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  WARNING: {r.stderr[:300]}")

    print(f"  torch/XLA install took {time.time()-t0:.1f}s")

    # Verify after install
    importlib.invalidate_caches()
    try:
        if "torch" in sys.modules:
            del sys.modules["torch"]
        if "torch_xla" in sys.modules:
            del sys.modules["torch_xla"]
        import torch
        import torch_xla.core.xla_model as xm
        dev = xm.xla_device()
        _ = torch.tensor([1.0], device=dev)
        torch_xla_ok = True
        print(f"  torch_xla NOW functional (torch={torch.__version__})")
    except Exception as e:
        print(f"  torch_xla still not working: {e}")
        print("  Continuing without TPU (will use GPU or CPU)...")

# ---------------------------------------------------------------------------
# 2.3  Install HuggingFace + HTML processing packages (single pip call)
# ---------------------------------------------------------------------------
print("\n  Installing HuggingFace + utility packages...")
t0 = time.time()

PACKAGES = [
    "datasets>=2.19.0",
    "pandas>=2.0.0",
    "transformers>=4.40.0",
    "huggingface-hub>=0.23.0",
    "accelerate>=0.30.0",
    "beautifulsoup4>=4.12.0",
    "lxml>=5.0.0",
    "tiktoken>=0.7.0",
]

# Single pip call: resolves dependencies once, 3-5x faster than sequential
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q"] + PACKAGES,
    capture_output=True, text=True,
)
if result.returncode == 0:
    print(f"  All packages installed in {time.time()-t0:.1f}s")
else:
    print(f"  Some packages may have failed ({time.time()-t0:.1f}s)")
    print(f"  stderr: {result.stderr[:300]}")
    # Fallback: install one by one to identify failures
    for pkg in PACKAGES:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            capture_output=True, text=True,
        )
        status = "OK" if r.returncode == 0 else "FAILED"
        print(f"    {pkg}: {status}")

# ---------------------------------------------------------------------------
# 2.4  Verify Critical Imports
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
    ("tiktoken", "tiktoken"),
]

for mod_name, display in CHECKS:
    try:
        mod = importlib.import_module(mod_name)
        ver = getattr(mod, "__version__", "?")
        print(f"    {display:20s} v{ver:15s} OK")
    except ImportError as e:
        print(f"    {display:20s} {'N/A':15s} MISSING")

# ---------------------------------------------------------------------------
# 2.5  Accelerator Compatibility Checks
# ---------------------------------------------------------------------------
print("\n  Accelerator compatibility checks...\n")

issues = []

# Check 1: XLA tensor ops
if torch_xla_ok:
    try:
        import torch
        import torch_xla.core.xla_model as xm
        dev = xm.xla_device()
        t = torch.tensor([1.0, 2.0, 3.0], device=dev)
        r = (t * t).sum().item()
        assert abs(r - 14.0) < 0.01
        print(f"    XLA tensor ops:       OK (1^2+2^2+3^2 = {r})")
    except Exception as e:
        issues.append(f"XLA tensor ops: {e}")
        print(f"    XLA tensor ops:       FAILED")

    # Check 2: bfloat16
    try:
        bf = torch.tensor([1.0], dtype=torch.bfloat16, device=dev)
        print(f"    bfloat16 support:     OK")
    except Exception as e:
        issues.append(f"bfloat16: {e}")
        print(f"    bfloat16 support:     FAILED")
else:
    print("    XLA tensor ops:       SKIPPED (no TPU)")
    print("    bfloat16 support:     SKIPPED (no TPU)")

# Check 3: Transformers tokenizer
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    ids = tok("test", return_tensors="pt").input_ids
    print(f"    Transformers tokenize: OK (vocab={tok.vocab_size})")
except Exception as e:
    issues.append(f"Transformers: {e}")
    print(f"    Transformers tokenize: FAILED ({e})")

# Check 4: GPU check
try:
    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        if torch_xla_ok:
            print(f"    GPU check:            NOTE (GPU '{gpu}' also available)")
        else:
            print(f"    GPU check:            GPU MODE ({gpu})")
    else:
        print(f"    GPU check:            No GPU (expected for TPU mode)")
except Exception:
    print(f"    GPU check:            OK")

# ---------------------------------------------------------------------------
# 2.6  Summary
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
if issues:
    print(f"  ISSUES: {len(issues)}")
    for i in issues:
        print(f"    - {i}")
else:
    if torch_xla_ok:
        print("  ALL TPU CHECKS PASSED")
    else:
        print("  TPU not available. Pipeline will use GPU or CPU.")

print("\n  TPU CELL 2 COMPLETE")
print("=" * 60)
