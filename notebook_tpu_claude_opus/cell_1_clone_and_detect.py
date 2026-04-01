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
# 1.3  Resolve Target Repository + Clone/Update
# ---------------------------------------------------------------------------
# Accept repository from environment variable so each user can choose their own repo.
# Supported formats:
#   - Full HTTPS URL:  https://github.com/<owner>/<repo>.git
#   - SSH URL:         git@github.com:<owner>/<repo>.git
#   - Owner/repo:      phanirishindra/pruning-html
#   - @owner/repo:     @phanirishindra/pruning-html
#
# Priority:
#   1) REPO_URL env var
#   2) REPO_SLUG env var
#   3) fallback default slug

DEFAULT_REPO_SLUG = "phanirishindra/pruning-html"
repo_input = (
    os.environ.get("REPO_URL")
    or os.environ.get("REPO_SLUG")
    or DEFAULT_REPO_SLUG
).strip()

def normalize_repo_to_https_git(value: str) -> tuple[str, str]:
    if value.startswith("@"):
        value = value[1:]

    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
        slug = value
        return f"https://github.com/{slug}.git", slug

    ssh_match = re.fullmatch(r"git@github\.com:([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(?:\.git)?", value)
    if ssh_match:
        slug = ssh_match.group(1)
        return f"https://github.com/{slug}.git", slug

    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"} and parsed.netloc.lower() == "github.com":
        path = parsed.path.strip("/")
        if path.endswith(".git"):
            path = path[:-4]
        if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", path):
            slug = path
            return f"https://github.com/{slug}.git", slug

    raise ValueError(
        "Invalid repository input. Use one of: "
        "'owner/repo', '@owner/repo', "
        "'https://github.com/owner/repo(.git)', "
        "or 'git@github.com:owner/repo(.git)'."
    )

REPO_URL, REPO_SLUG = normalize_repo_to_https_git(repo_input)
REPO_NAME = REPO_SLUG.split("/")[-1]

if is_colab:
    WORK_DIR = f"/content/{REPO_NAME}"
elif is_kaggle:
    WORK_DIR = f"/kaggle/working/{REPO_NAME}"
else:
    WORK_DIR = os.path.abspath(REPO_NAME)

print(f"\n  Repository input: {repo_input}")
print(f"  Normalized slug:  {REPO_SLUG}")
print(f"  Clone URL:        {REPO_URL}")
print(f"  Working dir:      {WORK_DIR}")

if os.path.exists(os.path.join(WORK_DIR, ".git")):
    print("  Repo exists. Syncing with remote...")

    current_origin = subprocess.run(
        ["git", "-C", WORK_DIR, "remote", "get-url", "origin"],
        capture_output=True, text=True, check=False
    ).stdout.strip()

    if current_origin and current_origin != REPO_URL:
        print("  Updating origin remote...")
        print(f"    from: {current_origin}")
        print(f"    to:   {REPO_URL}")
        subprocess.run(
            ["git", "-C", WORK_DIR, "remote", "set-url", "origin", REPO_URL],
            check=True
        )

    subprocess.run(["git", "-C", WORK_DIR, "fetch", "origin"], check=False)
    pull_result = subprocess.run(
        ["git", "-C", WORK_DIR, "pull", "--ff-only"],
        check=False
    )
    if pull_result.returncode != 0:
        print("  WARNING: --ff-only pull failed (possible divergence).")
        print("  You can resolve manually in the repo directory.")
else:
    print("  Cloning repository...")
    subprocess.run(["git", "clone", REPO_URL, WORK_DIR], check=True)
    print("  Clone complete.")

os.chdir(WORK_DIR)

# ---------------------------------------------------------------------------
# 1.4  Ensure timing_tracker module exists (self-healing fallback)
# ---------------------------------------------------------------------------
from pathlib import Path
import importlib.util
import textwrap

def _is_importable(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None

def _ensure_timing_tracker(work_dir: str) -> str:
    """
    Ensure `timing_tracker` is importable.
    Priority:
      1) Existing import path
      2) air_llm/timing_tracker.py (shim copy to notebook_cells_tpu)
      3) Generate new notebook_cells_tpu/timing_tracker.py
    Returns status string.
    """
    if _is_importable("timing_tracker"):
        return "existing"

    nb_tpu_dir = Path(work_dir) / "notebook_cells_tpu"
    air_llm_tracker = Path(work_dir) / "air_llm" / "timing_tracker.py"
    target_tracker = nb_tpu_dir / "timing_tracker.py"

    nb_tpu_dir.mkdir(parents=True, exist_ok=True)

    # Option 1: copy from air_llm if present
    if air_llm_tracker.exists():
        target_tracker.write_text(air_llm_tracker.read_text(encoding="utf-8"), encoding="utf-8")
        importlib.invalidate_caches()
        if _is_importable("timing_tracker"):
            return "copied_from_air_llm"

    # Option 2: generate robust fallback tracker
    tracker_code = textwrap.dedent(
        '''
        """
        Lightweight timing tracker for notebook / TPU workflows.
        Features:
          - context manager timing blocks
          - decorators for sync / async functions
          - step-level logging
          - summary table with totals
          - optional JSON export
          - safe fallback in all runtimes (Colab/Kaggle/Local)
        """
        from __future__ import annotations
        import time
        import json
        import threading
        from dataclasses import dataclass, asdict
        from typing import Callable, Optional, Any, Dict, List
        from contextlib import contextmanager

        @dataclass
        class TimingEvent:
            name: str
            start_ts: float
            end_ts: float
            duration_s: float
            meta: Optional[Dict[str, Any]] = None

        class TimingTracker:
            def __init__(self, name: str = "default", auto_print: bool = True):
                self.name = name
                self.auto_print = auto_print
                self._events: List[TimingEvent] = []
                self._active: Dict[str, float] = {}
                self._lock = threading.Lock()

            def start(self, step: str):
                now = time.perf_counter()
                with self._lock:
                    self._active[step] = now

            def stop(self, step: str, meta: Optional[Dict[str, Any]] = None) -> float:
                now = time.perf_counter()
                with self._lock:
                    if step not in self._active:
                        raise KeyError(f"Step '{step}' was not started.")
                    start = self._active.pop(step)
                    dur = now - start
                    self._events.append(TimingEvent(
                        name=step, start_ts=start, end_ts=now, duration_s=dur, meta=meta
                    ))
                if self.auto_print:
                    print(f"[timing] {step}: {dur:.3f}s")
                return dur

            @contextmanager
            def track(self, step: str, meta: Optional[Dict[str, Any]] = None):
                self.start(step)
                try:
                    yield
                finally:
                    self.stop(step, meta=meta)

            def log(self, step: str, duration_s: float, meta: Optional[Dict[str, Any]] = None):
                end = time.perf_counter()
                start = end - float(duration_s)
                with self._lock:
                    self._events.append(TimingEvent(
                        name=step, start_ts=start, end_ts=end,
                        duration_s=float(duration_s), meta=meta
                    ))
                if self.auto_print:
                    print(f"[timing] {step}: {duration_s:.3f}s (manual)")

            def total(self) -> float:
                return sum(e.duration_s for e in self._events)

            def events(self) -> List[TimingEvent]:
                return list(self._events)

            def summary(self, sort_desc: bool = True) -> str:
                rows = self.events()
                if sort_desc:
                    rows = sorted(rows, key=lambda e: e.duration_s, reverse=True)
                lines = []
                lines.append(f"Timing Summary [{self.name}]")
                lines.append("-" * 72)
                lines.append(f"{'Step':40} {'Seconds':>12} {'Pct':>8}")
                lines.append("-" * 72)
                total = self.total() or 1e-12
                for e in rows:
                    pct = (e.duration_s / total) * 100
                    lines.append(f"{e.name[:40]:40} {e.duration_s:12.3f} {pct:7.2f}%")
                lines.append("-" * 72)
                lines.append(f"{'TOTAL':40} {self.total():12.3f} {100.00:7.2f}%")
                text = "\\n".join(lines)
                return text

            def print_summary(self, sort_desc: bool = True):
                print(self.summary(sort_desc=sort_desc))

            def to_json(self, path: str):
                payload = {
                    "name": self.name,
                    "total_s": self.total(),
                    "events": [asdict(e) for e in self._events],
                }
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)

        # Global default tracker
        _DEFAULT_TRACKER = TimingTracker(name="global", auto_print=True)

        def get_tracker() -> TimingTracker:
            return _DEFAULT_TRACKER

        @contextmanager
        def track(step: str, meta: Optional[Dict[str, Any]] = None):
            with _DEFAULT_TRACKER.track(step, meta=meta):
                yield

        def timed(step_name: Optional[str] = None):
            def _decorator(fn: Callable):
                name = step_name or fn.__name__
                def _wrapped(*args, **kwargs):
                    with _DEFAULT_TRACKER.track(name):
                        return fn(*args, **kwargs)
                return _wrapped
            return _decorator
        '''
    ).strip() + "\n"

    target_tracker.write_text(tracker_code, encoding="utf-8")
    importlib.invalidate_caches()

    if _is_importable("timing_tracker"):
        return "generated_fallback"

    return "failed"

# Setup Python path first (required before timing_tracker/tpu_config imports)
for p in [
    WORK_DIR,
    os.path.join(WORK_DIR, "notebook_cells_tpu"),
    os.path.join(WORK_DIR, "air_llm"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Ensure timing_tracker is importable
tracker_status = _ensure_timing_tracker(WORK_DIR)
print(f"  timing_tracker status: {tracker_status}")

# Configure timing tracker with compatibility fallback
try:
    import timing_tracker as tt

    if hasattr(tt, "configure_tracker"):
        tracker = tt.configure_tracker(
            tracker_name="tpu_pipeline",
            repo_slug=REPO_SLUG,
            platform_name=platform_name,
            device_name="tpu" if tpu_available else "gpu_or_cpu",
            persist=True,
            artifacts_dir=os.path.join(WORK_DIR, "artifacts", "timing"),
            auto_print=True,
        )
        print("  timing_tracker: configured via configure_tracker")
        if hasattr(tracker, "run_id"):
            print(f"  run_id:         {tracker.run_id}")
    elif hasattr(tt, "get_tracker"):
        tracker = tt.get_tracker()
        print("  timing_tracker: configured via get_tracker fallback")
    else:
        print("  WARNING: timing_tracker has no configure_tracker/get_tracker")
except Exception as e:
    print(f"  WARNING: timing_tracker setup failed: {e}")


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
