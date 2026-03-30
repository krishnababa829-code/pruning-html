"""timing_tracker.py - Production timing infrastructure for TPU/GPU/CPU pipelines.

Provides:
  - TimingTracker:  hierarchical nested timing spans with JSONL persistence
  - PipelineTimer:  high-level wrapper for the full pipeline (phases + rows)
  - RowTimer:       per-row timing with ETA calculation
  - format_duration: human-readable duration strings
  - detect_layer_count: estimate transformer layer count from model ID
  - configure_tracker / get_tracker: global singleton helpers

All classes referenced by cells 1-7 are defined here.
"""
from __future__ import annotations

import json
import math
import os
import re
import socket
import time
import uuid
import threading
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================================
#  Utility functions (imported by cells 5, 6, 7)
# ============================================================================

def format_duration(seconds: float) -> str:
    """Convert seconds to a human-readable string like '2m 34.5s' or '1h 05m 12s'."""
    if seconds < 0:
        return f"-{format_duration(-seconds)}"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:04.1f}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins:02d}m {secs:04.1f}s"


def detect_layer_count(model_id: str) -> int:
    """Estimate the number of transformer layers from a model ID string.

    Uses known Qwen2.5 architecture specs. Falls back to a conservative
    estimate based on parameter count extracted from the model name.
    """
    model_lower = model_id.lower()

    # Known Qwen2.5 architectures
    known = {
        "0.5b": 24, "1.5b": 28, "3b": 36, "7b": 28,
        "14b": 40, "32b": 64, "72b": 80,
    }
    for size_tag, layers in known.items():
        if size_tag in model_lower:
            return layers

    # Fallback: extract number before 'b' and estimate
    m = re.search(r"(\d+\.?\d*)b", model_lower)
    if m:
        params_b = float(m.group(1))
        # Rough heuristic: layers ~ 4 * log2(params_billions) + 20
        return max(12, int(4 * math.log2(max(params_b, 0.5)) + 20))

    return 32  # safe default


# ============================================================================
#  TimingEvent dataclass
# ============================================================================

@dataclass
class TimingEvent:
    run_id: str
    tracker_name: str
    step: str
    parent_step: Optional[str]
    depth: int
    start_ts: float
    end_ts: float
    duration_s: float
    meta: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
#  TimingTracker - hierarchical timing with JSONL persistence
# ============================================================================

class TimingTracker:
    """Production timing tracker with nested spans, JSONL logs, and LLM metrics."""

    def __init__(
        self,
        tracker_name: str = "global",
        repo_slug: Optional[str] = None,
        platform_name: Optional[str] = None,
        device_name: Optional[str] = None,
        persist: bool = True,
        artifacts_dir: str = "artifacts/timing",
        auto_print: bool = True,
    ):
        self.tracker_name = tracker_name
        self.repo_slug = repo_slug or os.getenv("REPO_SLUG", "unknown/unknown")
        self.platform_name = platform_name or self._detect_platform()
        self.device_name = device_name or self._detect_device()
        self.persist = persist
        self.auto_print = auto_print

        self.run_id = self._new_run_id()
        self.run_started_at = time.perf_counter()
        self.run_started_at_iso = datetime.now(timezone.utc).isoformat()

        self._lock = threading.RLock()
        self._events: List[TimingEvent] = []
        self._stack: List[Dict[str, Any]] = []

        self.artifacts_dir = Path(artifacts_dir)
        if self.persist:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            self._jsonl_path = self.artifacts_dir / f"{self.run_id}.jsonl"
            self._write_run_header()
        else:
            self._jsonl_path = None

    # ---- internal helpers ----

    @staticmethod
    def _new_run_id() -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"run_{ts}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _detect_platform() -> str:
        if os.path.exists("/content") and not os.path.exists("/kaggle"):
            return "colab"
        if os.path.exists("/kaggle/working"):
            return "kaggle"
        return "local"

    @staticmethod
    def _detect_device() -> str:
        if os.getenv("COLAB_TPU_ADDR") or os.getenv("TPU_NAME"):
            return "tpu"
        try:
            import torch
            if torch.cuda.is_available():
                return f"gpu:{torch.cuda.get_device_name(0)}"
        except Exception:
            pass
        return "cpu"

    def _base_meta(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "tracker_name": self.tracker_name,
            "repo_slug": self.repo_slug,
            "platform": self.platform_name,
            "device": self.device_name,
            "host": socket.gethostname(),
            "started_at_utc": self.run_started_at_iso,
        }

    def _write_jsonl(self, record: Dict[str, Any]) -> None:
        if not self.persist or self._jsonl_path is None:
            return
        try:
            with self._jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass  # non-fatal: don't crash pipeline over logging I/O

    def _write_run_header(self) -> None:
        self._write_jsonl({"type": "run_header", **self._base_meta()})

    # ---- hierarchical timing spans ----

    def start(self, step: str, meta: Optional[Dict[str, Any]] = None) -> None:
        now = time.perf_counter()
        with self._lock:
            parent = self._stack[-1]["step"] if self._stack else None
            depth = len(self._stack)
            self._stack.append({
                "step": step, "start": now,
                "parent": parent, "depth": depth,
                "meta": meta or {},
            })

    def stop(self, step: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> float:
        now = time.perf_counter()
        with self._lock:
            if not self._stack:
                raise RuntimeError("No active timing span to stop.")
            active = self._stack.pop()
            if step is not None and active["step"] != step:
                raise RuntimeError(
                    f"Timing stop mismatch: expected '{active['step']}', got '{step}'."
                )
            duration = now - active["start"]
            merged_meta = {**self._base_meta(), **active.get("meta", {})}
            if meta:
                merged_meta.update(meta)
            event = TimingEvent(
                run_id=self.run_id, tracker_name=self.tracker_name,
                step=active["step"], parent_step=active["parent"],
                depth=active["depth"], start_ts=active["start"],
                end_ts=now, duration_s=duration, meta=merged_meta,
            )
            self._events.append(event)
            self._write_jsonl({"type": "timing_event", **asdict(event)})
        if self.auto_print:
            indent = "  " * event.depth
            print(f"[timing] {indent}{event.step}: {format_duration(event.duration_s)}")
        return duration

    @contextmanager
    def track(self, step: str, meta: Optional[Dict[str, Any]] = None):
        self.start(step, meta=meta)
        try:
            yield
        finally:
            self.stop(step=step)

    # ---- LLM-specific metrics ----

    def log_inference_metrics(
        self, step: str, *,
        ttft_s: Optional[float] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        wall_time_s: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        calc_total = total_tokens
        if calc_total is None and prompt_tokens is not None and completion_tokens is not None:
            calc_total = prompt_tokens + completion_tokens
        tps = None
        if wall_time_s and completion_tokens:
            effective = wall_time_s - (ttft_s or 0)
            tps = completion_tokens / max(effective, 1e-9)
        payload = {
            "type": "inference_metrics", **self._base_meta(),
            "step": step, "ttft_s": ttft_s,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": calc_total,
            "wall_time_s": wall_time_s, "tokens_per_s": tps,
        }
        if meta:
            payload["meta"] = meta
        self._write_jsonl(payload)

    # ---- summaries / exports ----

    def events(self) -> List[TimingEvent]:
        with self._lock:
            return list(self._events)

    def total_wall_s(self) -> float:
        return time.perf_counter() - self.run_started_at

    def total_timed_s(self) -> float:
        with self._lock:
            return sum(e.duration_s for e in self._events)

    def summary_text(self, sort_desc: bool = True) -> str:
        evs = self.events()
        if sort_desc:
            evs = sorted(evs, key=lambda e: e.duration_s, reverse=True)
        total = sum(e.duration_s for e in evs) or 1e-9
        lines = [
            f"Timing Summary [{self.tracker_name}] run_id={self.run_id}",
            "-" * 96,
            f"{'Step':45} {'Depth':>5} {'Seconds':>12} {'Pct':>8} {'Parent':20}",
            "-" * 96,
        ]
        for e in evs:
            pct = (e.duration_s / total) * 100.0
            lines.append(
                f"{e.step[:45]:45} {e.depth:5d} {e.duration_s:12.3f} {pct:7.2f}% "
                f"{(e.parent_step or '-')[:20]:20}"
            )
        lines.append("-" * 96)
        lines.append(f"{'TOTAL_TIMED':45} {'':>5} {self.total_timed_s():12.3f} {100.00:7.2f}%")
        lines.append(f"{'TOTAL_WALL':45} {'':>5} {self.total_wall_s():12.3f}")
        return "\n".join(lines)

    def print_summary(self, sort_desc: bool = True) -> None:
        print(self.summary_text(sort_desc=sort_desc))

    def to_json(self, path: str) -> None:
        payload = {
            "run_id": self.run_id, "tracker_name": self.tracker_name,
            "base_meta": self._base_meta(),
            "total_wall_s": self.total_wall_s(),
            "total_timed_s": self.total_timed_s(),
            "events": [asdict(e) for e in self.events()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    def to_pandas(self):
        import pandas as pd
        rows = []
        for e in self.events():
            row = asdict(e)
            for k, v in (e.meta or {}).items():
                row[f"meta.{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows)


# ============================================================================
#  RowTimer - per-row timing with ETA
# ============================================================================

class RowTimer:
    """Tracks per-row inference timing and computes rolling ETA."""

    def __init__(self, total_rows: int):
        self.total_rows = total_rows
        self._row_times: List[float] = []
        self._current_start: Optional[float] = None
        self._skipped: int = 0
        self._generation_start: Optional[float] = None

    def start_generation_phase(self) -> None:
        self._generation_start = time.perf_counter()

    def start_row(self, idx: int) -> None:
        self._current_start = time.perf_counter()

    def end_row(self, idx: int, output_tokens: int = 0) -> float:
        if self._current_start is None:
            return 0.0
        elapsed = time.perf_counter() - self._current_start
        self._row_times.append(elapsed)
        self._current_start = None

        done = len(self._row_times) + self._skipped
        remaining = self.total_rows - done
        if self._row_times:
            avg = sum(self._row_times) / len(self._row_times)
            eta = avg * remaining
            print(f"    Row time: {format_duration(elapsed)} | "
                  f"Avg: {format_duration(avg)} | "
                  f"ETA: {format_duration(eta)} | "
                  f"Done: {done}/{self.total_rows}")
        return elapsed

    def skip_row(self, idx: int) -> None:
        self._skipped += 1

    def total_generation_time(self) -> float:
        if self._generation_start is None:
            return sum(self._row_times)
        return time.perf_counter() - self._generation_start


# ============================================================================
#  PipelineTimer - high-level pipeline orchestrator
# ============================================================================

class PipelineTimer:
    """Orchestrates timing for the full pipeline: phases + row-level tracking.

    Usage:
        timer = PipelineTimer(total_rows=100, total_layers=28)
        timer.start()
        timer.start_phase("model_loading")
        ...load model...
        timer.end_phase("model_loading")
        timer.start_phase("generation")
        timer.row_timer.start_generation_phase()
        for row in rows:
            timer.row_timer.start_row(i)
            ...generate...
            timer.row_timer.end_row(i, output_tokens=n)
        timer.end_phase("generation")
        timer.finish()
    """

    def __init__(
        self,
        total_rows: int = 100,
        total_layers: int = 28,
        report_path: Optional[str] = None,
    ):
        self.total_rows = total_rows
        self.total_layers = total_layers
        self.report_path = report_path
        self.row_timer = RowTimer(total_rows)

        self._pipeline_start: Optional[float] = None
        self._phases: Dict[str, Dict[str, Any]] = {}
        self._phase_order: List[str] = []

    def start(self) -> None:
        self._pipeline_start = time.perf_counter()
        print(f"  Pipeline timer started (rows={self.total_rows}, layers={self.total_layers})")

    def start_phase(self, name: str) -> None:
        self._phases[name] = {"start": time.perf_counter(), "end": None, "duration": None}
        if name not in self._phase_order:
            self._phase_order.append(name)

    def end_phase(self, name: str) -> float:
        if name not in self._phases:
            return 0.0
        now = time.perf_counter()
        p = self._phases[name]
        p["end"] = now
        p["duration"] = now - p["start"]
        print(f"  Phase '{name}' completed in {format_duration(p['duration'])}")
        return p["duration"]

    def finish(self) -> None:
        if self._pipeline_start is None:
            print("  WARNING: Pipeline timer was never started.")
            return

        total = time.perf_counter() - self._pipeline_start
        print("\n" + "=" * 60)
        print("  PIPELINE TIMING REPORT")
        print("=" * 60)
        for name in self._phase_order:
            p = self._phases.get(name, {})
            dur = p.get("duration", 0) or 0
            pct = (dur / total * 100) if total > 0 else 0
            print(f"  {name:35s} {format_duration(dur):>12s}  ({pct:5.1f}%)")
        print("-" * 60)
        print(f"  {'TOTAL':35s} {format_duration(total):>12s}  (100.0%)")
        print("=" * 60)

        # Save JSON report
        if self.report_path:
            try:
                os.makedirs(os.path.dirname(self.report_path) or ".", exist_ok=True)
                report = {
                    "total_s": total,
                    "total_human": format_duration(total),
                    "total_rows": self.total_rows,
                    "total_layers": self.total_layers,
                    "phases": {
                        name: {
                            "duration_s": (self._phases[name].get("duration") or 0),
                            "duration_human": format_duration(self._phases[name].get("duration") or 0),
                        }
                        for name in self._phase_order
                    },
                }
                with open(self.report_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
                print(f"  Report saved: {self.report_path}")
            except Exception as e:
                print(f"  WARNING: Could not save report: {e}")


# ============================================================================
#  Global singleton helpers
# ============================================================================

_DEFAULT_TRACKER: Optional[TimingTracker] = None


def configure_tracker(
    tracker_name: str = "global",
    repo_slug: Optional[str] = None,
    platform_name: Optional[str] = None,
    device_name: Optional[str] = None,
    persist: bool = True,
    artifacts_dir: str = "artifacts/timing",
    auto_print: bool = True,
) -> TimingTracker:
    global _DEFAULT_TRACKER
    _DEFAULT_TRACKER = TimingTracker(
        tracker_name=tracker_name, repo_slug=repo_slug,
        platform_name=platform_name, device_name=device_name,
        persist=persist, artifacts_dir=artifacts_dir,
        auto_print=auto_print,
    )
    return _DEFAULT_TRACKER


def get_tracker() -> TimingTracker:
    global _DEFAULT_TRACKER
    if _DEFAULT_TRACKER is None:
        _DEFAULT_TRACKER = TimingTracker()
    return _DEFAULT_TRACKER


@contextmanager
def track(step: str, meta: Optional[Dict[str, Any]] = None):
    t = get_tracker()
    with t.track(step, meta=meta):
        yield
