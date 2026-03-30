#!/usr/bin/env python3
"""
=============================================================================
  TPU CELL 7: SAVE FINAL CSVs & TIMING REPORT
=============================================================================
  - Enforces 8000-token budget per row
  - Saves newdataset-2-tpu.csv [page_id, html, response]
  - Generates timing report
  - Provides download instructions
=============================================================================
"""

import os
import sys
import csv
import json
import time
from pathlib import Path

import pandas as pd

for _p in [
    os.path.dirname(os.path.abspath(__file__)),
    os.path.join(os.getcwd(), "notebook_cells_tpu"),
    ".",
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tpu_config__1_ import (
    OUTPUT_FILE_1, OUTPUT_FILE_2, CHECKPOINT_FILE,
    MAX_TOKENS_PER_ROW, HTML_BUDGET_RATIO, MODEL_ID,
    IS_COLAB, IS_KAGGLE,
)
from timing_tracker import format_duration

print("\n" + "=" * 60)
print("  TPU CELL 7: SAVE & REPORT")
print("=" * 60)

# ---------------------------------------------------------------------------
# 7.1  Prerequisites
# ---------------------------------------------------------------------------
_missing = []
for _var in ["df", "responses", "pipeline_timer", "counting_tokenizer"]:
    if _var not in dir() and _var not in globals():
        _missing.append(_var)

if _missing:
    print(f"  ERROR: Missing: {_missing}. Run Cells 3, 5, 6 first.")
    raise RuntimeError(f"Missing: {_missing}")

print(f"  DataFrame: {len(df)} rows | Responses: {len(responses)}")

# ---------------------------------------------------------------------------
# 7.2  Shared Helpers (same as Cell 6, kept here for standalone execution)
# ---------------------------------------------------------------------------
def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(counting_tokenizer.encode(str(text), add_special_tokens=False))


def truncate_text(text: str, max_tok: int) -> str:
    ids = counting_tokenizer.encode(str(text), add_special_tokens=False)
    if len(ids) <= max_tok:
        return text
    return counting_tokenizer.decode(ids[:max_tok], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# 7.3  Attach Responses & Enforce Token Budget
# ---------------------------------------------------------------------------
pipeline_timer.start_phase("token_budget_enforcement")

df["response"] = df.index.map(lambda i: responses.get(i, "[ERROR] No response"))

truncated = 0
for row in df.itertuples():
    idx = row.Index
    overhead = count_tokens(str(df.at[idx, "page_id"])) + 20
    budget = MAX_TOKENS_PER_ROW - overhead
    h_tok = count_tokens(df.at[idx, "html"])
    r_tok = count_tokens(df.at[idx, "response"])
    if h_tok + r_tok > budget:
        h_bud = int(budget * HTML_BUDGET_RATIO)
        r_bud = budget - h_bud
        df.at[idx, "html"] = truncate_text(df.at[idx, "html"], h_bud)
        df.at[idx, "response"] = truncate_text(df.at[idx, "response"], r_bud)
        truncated += 1

print(f"  Truncated {truncated} rows to {MAX_TOKENS_PER_ROW}-token budget")
pipeline_timer.end_phase("token_budget_enforcement")

# ---------------------------------------------------------------------------
# 7.4  Save CSV
# ---------------------------------------------------------------------------
pipeline_timer.start_phase("save_dataset_2")
os.makedirs(os.path.dirname(OUTPUT_FILE_2) or ".", exist_ok=True)
df.to_csv(OUTPUT_FILE_2, index=False, quoting=csv.QUOTE_ALL)
pipeline_timer.end_phase("save_dataset_2")

print(f"  Saved: {OUTPUT_FILE_2} ({os.path.getsize(OUTPUT_FILE_2):,} bytes)")
print(f"  Columns: {list(df.columns)}")

# ---------------------------------------------------------------------------
# 7.5  Token Stats
# ---------------------------------------------------------------------------
tok_counts = []
for row in df.itertuples():
    tok_counts.append(
        count_tokens(str(row.page_id))
        + count_tokens(row.html)
        + count_tokens(row.response)
    )

if tok_counts:
    print(f"\n  Token stats: min={min(tok_counts):,} max={max(tok_counts):,} "
          f"mean={sum(tok_counts)//len(tok_counts):,}")
    over = sum(1 for t in tok_counts if t > MAX_TOKENS_PER_ROW)
    print(f"  Over budget: {over}")

# ---------------------------------------------------------------------------
# 7.6  Cleanup & Report
# ---------------------------------------------------------------------------
ckpt = Path(CHECKPOINT_FILE)
if ckpt.exists():
    ckpt.unlink()
    print("  Checkpoint cleaned up")

print("\n" + "=" * 60)
pipeline_timer.finish()

# ---------------------------------------------------------------------------
# 7.7  Output Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  FINAL OUTPUT FILES")
print("=" * 60)

for f in [OUTPUT_FILE_1, OUTPUT_FILE_2, TIMING_REPORT_FILE]:
    fp = f if os.path.isabs(f) else os.path.join(os.getcwd(), f)
    if os.path.exists(fp):
        print(f"  {os.path.basename(fp):35s} {os.path.getsize(fp):>12,} bytes")
    elif os.path.exists(f):
        print(f"  {os.path.basename(f):35s} {os.path.getsize(f):>12,} bytes")
    else:
        print(f"  {os.path.basename(f):35s} NOT FOUND")

print(f"\n  newdataset-1-tpu.csv: [page_id, html]")
print(f"  newdataset-2-tpu.csv: [page_id, html, response]")

if IS_COLAB:
    print("\n  Download (Colab):")
    print("    from google.colab import files")
    print(f"    files.download('{OUTPUT_FILE_2}')")
elif IS_KAGGLE:
    print("\n  Download (Kaggle): Save Version > Save & Run All")

print("\n  ALL CELLS COMPLETE")
print("=" * 60)
