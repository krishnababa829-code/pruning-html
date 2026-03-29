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

for _p in ["notebook_cells_tpu", os.path.join(os.getcwd(), "notebook_cells_tpu"), "."]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tpu_config import (
    OUTPUT_FILE_1, OUTPUT_FILE_2, CHECKPOINT_FILE,
    MAX_TOKENS_PER_ROW, HTML_BUDGET_RATIO, MODEL_ID,
    IS_COLAB, IS_KAGGLE,
)

for _p in [os.getcwd(), os.path.dirname(os.getcwd())]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from timing_tracker import format_duration

print("\n" + "=" * 60)
print("  TPU CELL 7: SAVE & REPORT")
print("=" * 60)

# ---------------------------------------------------------------------------
# 7.1  Prerequisites
# ---------------------------------------------------------------------------
_missing = []
for _var in ["df", "responses", "pipeline_timer", "counting_tokenizer"]:
    try: eval(_var)
    except NameError: _missing.append(_var)

if _missing:
    print(f"  ERROR: Missing: {_missing}. Run Cells 3, 5, 6 first.")
    raise RuntimeError(f"Missing: {_missing}")

print(f"  DataFrame: {len(df)} rows | Responses: {len(responses)}")

# ---------------------------------------------------------------------------
# 7.2  Attach Responses & Enforce Token Budget
# ---------------------------------------------------------------------------
pipeline_timer.start_phase("token_budget_enforcement")

df["response"] = df.index.map(lambda i: responses.get(i, "[ERROR] No response"))

def count_tokens(text):
    if not text: return 0
    return len(counting_tokenizer.encode(text, add_special_tokens=False))

def truncate_text(text, max_tok):
    ids = counting_tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tok: return text
    return counting_tokenizer.decode(ids[:max_tok], skip_special_tokens=True)

truncated = 0
for idx in df.index:
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
# 7.3  Save CSV
# ---------------------------------------------------------------------------
pipeline_timer.start_phase("save_dataset_2")
os.makedirs(os.path.dirname(OUTPUT_FILE_2) or ".", exist_ok=True)
df.to_csv(OUTPUT_FILE_2, index=False, quoting=csv.QUOTE_ALL)
pipeline_timer.end_phase("save_dataset_2")

print(f"  Saved: {OUTPUT_FILE_2} ({os.path.getsize(OUTPUT_FILE_2):,} bytes)")
print(f"  Columns: {list(df.columns)}")

# ---------------------------------------------------------------------------
# 7.4  Token Stats
# ---------------------------------------------------------------------------
tok_counts = []
for _, row in df.iterrows():
    tok_counts.append(count_tokens(str(row["page_id"])) + count_tokens(row["html"]) + count_tokens(row["response"]))

if tok_counts:
    print(f"\n  Token stats: min={min(tok_counts):,} max={max(tok_counts):,} mean={sum(tok_counts)//len(tok_counts):,}")
    print(f"  Over budget: {sum(1 for t in tok_counts if t > MAX_TOKENS_PER_ROW)}")

# ---------------------------------------------------------------------------
# 7.5  Cleanup & Report
# ---------------------------------------------------------------------------
ckpt = Path(CHECKPOINT_FILE)
if ckpt.exists():
    ckpt.unlink()
    print("  Checkpoint cleaned up")

print("\n" + "=" * 60)
pipeline_timer.finish()

# ---------------------------------------------------------------------------
# 7.6  Output Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  FINAL OUTPUT FILES")
print("=" * 60)

for f in [OUTPUT_FILE_1, OUTPUT_FILE_2, "timing_report_tpu.json"]:
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

print("\n  ALL TPU CELLS COMPLETE")
print("=" * 60)
