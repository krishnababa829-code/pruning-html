#!/usr/bin/env python3
"""
=============================================================================
  CELL 7: SAVE FINAL CSVs & TIMING REPORT
=============================================================================
  - Enforces 8000-token budget per row (40% HTML / 60% response)
  - Saves newdataset-2.csv with columns: [page_id, html, response]
  - Generates and displays full timing report
  - Exports timing_report.json
  - Cleans up checkpoint file
=============================================================================
"""

import os
import sys
import csv
import json
import time
from pathlib import Path

import pandas as pd

try:
    from notebook_config import (
        OUTPUT_FILE_2, CHECKPOINT_FILE, TIMING_REPORT_FILE,
        MAX_TOKENS_PER_ROW, HTML_BUDGET_RATIO, MODEL_ID,
    )
except ImportError:
    sys.path.insert(0, os.path.join(os.getcwd(), "notebook_cells"))
    from notebook_config import (
        OUTPUT_FILE_2, CHECKPOINT_FILE, TIMING_REPORT_FILE,
        MAX_TOKENS_PER_ROW, HTML_BUDGET_RATIO, MODEL_ID,
    )

sys.path.insert(0, os.getcwd())
from timing_tracker import format_duration

print("\n" + "=" * 60)
print("  CELL 7: SAVE FINAL CSVs & TIMING REPORT")
print("=" * 60)

# ---------------------------------------------------------------------------
# 7.1  Verify Prerequisites
# ---------------------------------------------------------------------------
print("\n  Checking prerequisites...")

_missing = []
for _var in ["df", "responses", "pipeline_timer", "counting_tokenizer"]:
    try:
        eval(_var)
    except NameError:
        _missing.append(_var)

if _missing:
    print(f"  ERROR: Missing variables: {_missing}")
    print("  Run cells in order: Cell 3 (dataset) > Cell 5 (model) > Cell 6 (inference) > Cell 7")
    print("  If runtime was restarted, re-run all cells from Cell 1.")
    raise RuntimeError(f"Missing: {_missing}. Run previous cells first.")

print(f"  DataFrame:       {len(df)} rows")
print(f"  Responses:       {len(responses)} generated")
print(f"  Pipeline timer:  active")

# ---------------------------------------------------------------------------
# 7.2  Attach Responses to DataFrame
# ---------------------------------------------------------------------------
print("\n  Attaching responses to DataFrame...")

pipeline_timer.start_phase("token_budget_enforcement")

df["response"] = df.index.map(lambda i: responses.get(i, "[ERROR] No response"))

print(f"  Columns: {list(df.columns)}")
print(f"  Rows with responses: {(df['response'] != '[ERROR] No response').sum()}")

# ---------------------------------------------------------------------------
# 7.3  Enforce Token Budget (8000 tokens per row)
# ---------------------------------------------------------------------------
print(f"\n  Enforcing {MAX_TOKENS_PER_ROW}-token budget per row...")
print(f"  Split: {int(HTML_BUDGET_RATIO*100)}% HTML / {int((1-HTML_BUDGET_RATIO)*100)}% response")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(counting_tokenizer.encode(text, add_special_tokens=False))


def truncate_text(text: str, max_tokens: int) -> str:
    ids = counting_tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return counting_tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


truncated_rows = 0
for idx in df.index:
    overhead = count_tokens(str(df.at[idx, "page_id"])) + 20
    budget = MAX_TOKENS_PER_ROW - overhead

    html_tokens = count_tokens(df.at[idx, "html"])
    resp_tokens = count_tokens(df.at[idx, "response"])

    if html_tokens + resp_tokens > budget:
        html_budget = int(budget * HTML_BUDGET_RATIO)
        resp_budget = budget - html_budget

        df.at[idx, "html"] = truncate_text(df.at[idx, "html"], html_budget)
        df.at[idx, "response"] = truncate_text(df.at[idx, "response"], resp_budget)
        truncated_rows += 1

print(f"  Truncated {truncated_rows} rows to fit budget")

pipeline_timer.end_phase("token_budget_enforcement")

# ---------------------------------------------------------------------------
# 7.4  Save newdataset-2.csv
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_FILE_2) if os.path.dirname(OUTPUT_FILE_2) else ".", exist_ok=True)
print(f"\n  Saving {OUTPUT_FILE_2}...")

pipeline_timer.start_phase("save_dataset_2")

df.to_csv(OUTPUT_FILE_2, index=False, quoting=csv.QUOTE_ALL)
file_size = os.path.getsize(OUTPUT_FILE_2)

pipeline_timer.end_phase("save_dataset_2")

print(f"  Saved: {OUTPUT_FILE_2}")
print(f"  Size: {file_size:,} bytes")
print(f"  Columns: {list(df.columns)}")
print(f"  Rows: {len(df)}")

# ---------------------------------------------------------------------------
# 7.5  Token Distribution Stats
# ---------------------------------------------------------------------------
print("\n  Token Distribution:")

token_counts = []
for _, row in df.iterrows():
    total = (
        count_tokens(str(row["page_id"]))
        + count_tokens(row["html"])
        + count_tokens(row["response"])
    )
    token_counts.append(total)

if token_counts:
    print(f"    Min tokens/row:  {min(token_counts):,}")
    print(f"    Max tokens/row:  {max(token_counts):,}")
    print(f"    Mean tokens/row: {sum(token_counts)//len(token_counts):,}")
    print(f"    Total tokens:    {sum(token_counts):,}")
    over_budget = sum(1 for t in token_counts if t > MAX_TOKENS_PER_ROW)
    print(f"    Over budget:     {over_budget} rows")

# ---------------------------------------------------------------------------
# 7.6  Cleanup Checkpoint
# ---------------------------------------------------------------------------
checkpoint_path = Path(CHECKPOINT_FILE)
if checkpoint_path.exists():
    checkpoint_path.unlink()
    print("\n  Checkpoint file cleaned up")

# ---------------------------------------------------------------------------
# 7.7  Finalize Timing Report
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  TIMING REPORT")
print("=" * 60)

pipeline_timer.finish()

# ---------------------------------------------------------------------------
# 7.8  Output File Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  FINAL OUTPUT FILES")
print("=" * 60)

from notebook_config import OUTPUT_FILE_1

for fpath in [OUTPUT_FILE_1, OUTPUT_FILE_2, "timing_report.json"]:
    if os.path.exists(fpath):
        fsize = os.path.getsize(fpath)
        print(f"  {fpath:30s} {fsize:>12,} bytes")
    else:
        print(f"  {fpath:30s} NOT FOUND")

print(f"\n  newdataset-1.csv columns: [page_id, html]")
print(f"  newdataset-2.csv columns: [page_id, html, response]")

# ---------------------------------------------------------------------------
# 7.9  Download Instructions
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)

try:
    from notebook_config import IS_COLAB, IS_KAGGLE
except ImportError:
    IS_COLAB = IS_KAGGLE = False

if IS_COLAB:
    print("  DOWNLOAD FILES (Colab):")
    print("  Run this in a new cell:")
    print("    from google.colab import files")
    print(f"    files.download('{OUTPUT_FILE_1}')")
    print(f"    files.download('{OUTPUT_FILE_2}')")
    print(f"    files.download('timing_report.json')")
elif IS_KAGGLE:
    print("  DOWNLOAD FILES (Kaggle):")
    print(f"  Files are in: /kaggle/working/")
    print("  Click 'Save Version' > 'Save & Run All' to persist output.")
else:
    print("  Files saved to current directory.")

print("\n  ALL CELLS COMPLETE")
print("=" * 60)
