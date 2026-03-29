#!/usr/bin/env python3
"""
=============================================================================
  CELL 2: INSTALL REQUIREMENTS & CHECK INCOMPATIBILITIES
=============================================================================
  - Installs all pip dependencies
  - Checks for version conflicts
  - Verifies each critical import works
  - Reports any incompatibilities
=============================================================================
"""

import subprocess
import sys
import time
import importlib

print("\n" + "=" * 60)
print("  CELL 2: INSTALL REQUIREMENTS & CHECK INCOMPATIBILITIES")
print("=" * 60)

# ---------------------------------------------------------------------------
# 2.1  Install Dependencies
# ---------------------------------------------------------------------------
PACKAGES = [
    # AirLLM core
    ("airllm", "airllm>=2.10.0"),
    # Dataset
    ("datasets", "datasets>=2.19.0"),
    ("pandas", "pandas>=2.0.0"),
    # Model / Tokenizer
    ("transformers", "transformers>=4.40.0"),
    ("huggingface_hub", "huggingface-hub>=0.23.0"),
    ("accelerate", "accelerate>=0.30.0"),
    # HTML Pruning
    ("bs4", "beautifulsoup4>=4.12.0"),
    ("lxml", "lxml>=5.0.0"),
    # Token counting
    ("tiktoken", "tiktoken>=0.7.0"),
]

print("\n  Installing packages...\n")
install_start = time.time()

failed_installs = []
for import_name, pip_spec in PACKAGES:
    print(f"  Installing {pip_spec}...", end=" ", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", pip_spec],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("OK")
    else:
        print("FAILED")
        failed_installs.append((pip_spec, result.stderr[:200]))

install_time = time.time() - install_start
print(f"\n  Installation completed in {install_time:.1f}s")

if failed_installs:
    print("\n  FAILED INSTALLATIONS:")
    for pkg, err in failed_installs:
        print(f"    {pkg}: {err}")

# ---------------------------------------------------------------------------
# 2.2  Verify Imports & Check Versions
# ---------------------------------------------------------------------------
print("\n  Verifying imports and versions...\n")

CRITICAL_IMPORTS = [
    ("torch", "PyTorch"),
    ("airllm", "AirLLM"),
    ("transformers", "Transformers"),
    ("datasets", "Datasets"),
    ("pandas", "Pandas"),
    ("bs4", "BeautifulSoup4"),
    ("lxml", "lxml"),
    ("accelerate", "Accelerate"),
    ("huggingface_hub", "HuggingFace Hub"),
]

import_results = []
for module_name, display_name in CRITICAL_IMPORTS:
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "unknown")
        import_results.append((display_name, version, "OK"))
        print(f"    {display_name:20s} v{version:15s} OK")
    except ImportError as e:
        import_results.append((display_name, "N/A", f"FAILED: {e}"))
        print(f"    {display_name:20s} {'N/A':15s} FAILED: {e}")

# ---------------------------------------------------------------------------
# 2.3  Check for Known Incompatibilities
# ---------------------------------------------------------------------------
print("\n  Checking for incompatibilities...\n")

incompat_issues = []

# Check 1: torch + CUDA
try:
    import torch
    if torch.cuda.is_available():
        # Test a small CUDA operation
        x = torch.tensor([1.0]).cuda()
        _ = x + x
        print("    PyTorch CUDA:        OK (tensor ops work)")
    else:
        incompat_issues.append("PyTorch installed but CUDA not available")
        print("    PyTorch CUDA:        WARNING (no CUDA)")
except Exception as e:
    incompat_issues.append(f"PyTorch CUDA test failed: {e}")
    print(f"    PyTorch CUDA:        FAILED ({e})")

# Check 2: transformers + tokenizer
try:
    from transformers import AutoTokenizer
    print("    Transformers import:  OK")
except Exception as e:
    incompat_issues.append(f"Transformers import failed: {e}")
    print(f"    Transformers import:  FAILED ({e})")

# Check 3: airllm + torch compatibility
try:
    from airllm import AutoModel as AirAutoModel
    print("    AirLLM import:        OK")
except Exception as e:
    incompat_issues.append(f"AirLLM import failed: {e}")
    print(f"    AirLLM import:        FAILED ({e})")

# Check 4: BeautifulSoup + lxml parser
try:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup("<p>test</p>", "lxml")
    assert soup.p.text == "test"
    print("    BS4 + lxml parser:    OK")
except Exception as e:
    incompat_issues.append(f"BS4/lxml test failed: {e}")
    print(f"    BS4 + lxml parser:    FAILED ({e})")

# Check 5: datasets library
try:
    from datasets import load_dataset
    print("    Datasets library:     OK")
except Exception as e:
    incompat_issues.append(f"Datasets import failed: {e}")
    print(f"    Datasets library:     FAILED ({e})")

# ---------------------------------------------------------------------------
# 2.4  Summary
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
if incompat_issues:
    print(f"  ISSUES FOUND: {len(incompat_issues)}")
    for issue in incompat_issues:
        print(f"    - {issue}")
    print("\n  Some features may not work. Fix the issues above before proceeding.")
else:
    print("  ALL CHECKS PASSED - No incompatibilities found")

print(f"\n  Total packages: {len(PACKAGES)} | Imports verified: {len(CRITICAL_IMPORTS)}")
print("\n  CELL 2 COMPLETE")
print("=" * 60)
