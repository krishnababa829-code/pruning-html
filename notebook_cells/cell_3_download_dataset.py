#!/usr/bin/env python3
"""
=============================================================================
  CELL 3: DOWNLOAD DATASET & PRUNE HTML
=============================================================================
  - Downloads N rows from HuggingFace (user configurable via ROW_COUNT)
  - Keeps ONLY page_id and html columns
  - Prunes invisible HTML tags (script, style, svg, comments, data-*)
  - Saves newdataset-1.csv with columns: [page_id, html]
  - Shows dataset statistics
=============================================================================
"""

import os
import re
import csv
import sys
import time

import pandas as pd

# Import config (handles Colab/Kaggle/Local paths)
# Try multiple path strategies to handle any execution context
for _p in ["notebook_cells", os.path.join(os.getcwd(), "notebook_cells"), "."]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from notebook_config import (
    ROW_COUNT, SOURCE_DATASET, OUTPUT_FILE_1,
    WORK_DIR, IS_COLAB, IS_KAGGLE,
)

print("\n" + "=" * 60)
print("  CELL 3: DOWNLOAD DATASET & PRUNE HTML")
print(f"  Rows requested: {ROW_COUNT}")
print(f"  Source: {SOURCE_DATASET}")
print("=" * 60)

# ---------------------------------------------------------------------------
# 3.1  Download from HuggingFace
# ---------------------------------------------------------------------------
print("\n  Downloading dataset...")
download_start = time.time()

from datasets import load_dataset

ds = load_dataset(SOURCE_DATASET, split=f"train[:{ROW_COUNT}]")
df = ds.to_pandas()

download_time = time.time() - download_start
print(f"  Downloaded in {download_time:.1f}s")
print(f"  Raw shape: {df.shape}")
print(f"  Raw columns: {list(df.columns)}")

# ---------------------------------------------------------------------------
# 3.2  Extract ONLY page_id and html columns
# ---------------------------------------------------------------------------
print("\n  Detecting page_id and html columns...")

col_map = {}
for col in df.columns:
    lower = col.lower().strip()
    if "page" in lower and "id" in lower:
        col_map["page_id"] = col
    if lower == "html":
        col_map["html"] = col

if "page_id" not in col_map or "html" not in col_map:
    print(f"  WARNING: Auto-detect failed. Available columns: {list(df.columns)}")
    print(f"  Falling back to first two columns.")
    col_map["page_id"] = df.columns[0]
    col_map["html"] = df.columns[1]

print(f"  Mapped: page_id='{col_map['page_id']}', html='{col_map['html']}'")

# Keep ONLY these two columns
df = df[[col_map["page_id"], col_map["html"]]].copy()
df.columns = ["page_id", "html"]

print(f"  Final columns: {list(df.columns)}")
print(f"  Rows: {len(df)}")

# ---------------------------------------------------------------------------
# 3.3  Prune Invisible HTML
# ---------------------------------------------------------------------------
print("\n  Pruning invisible HTML tags...")
prune_start = time.time()

from bs4 import BeautifulSoup, Comment

INVISIBLE_TAGS = frozenset([
    "style", "script", "noscript", "meta", "link",
    "iframe", "object", "embed", "svg", "path", "head",
])

STRIP_ATTRS = frozenset([
    "class", "style", "onclick", "onload", "onmouseover",
    "onfocus", "onblur", "onchange", "onsubmit", "onerror",
    "onkeydown", "onkeyup", "onkeypress",
])


def prune_html(raw_html: str) -> str:
    """Strip invisible tags, keep structural HTML."""
    if not raw_html or not isinstance(raw_html, str):
        return ""

    soup = BeautifulSoup(raw_html, "lxml")

    # Remove invisible tags
    for tag_name in INVISIBLE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # Strip noisy attributes
    for tag in soup.find_all(True):
        removable = [
            attr for attr in tag.attrs
            if attr in STRIP_ATTRS or attr.startswith("data-")
        ]
        for attr in removable:
            del tag[attr]

    # Remove empty non-self-closing tags
    self_closing = frozenset(["br", "hr", "img", "input"])
    for tag in soup.find_all(True):
        if (
            tag.name not in self_closing
            and not tag.get_text(strip=True)
            and not tag.find_all(["img", "br", "hr", "input", "table"])
        ):
            tag.decompose()

    # Collapse whitespace
    cleaned = str(soup)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


# Track sizes for stats
original_sizes = df["html"].str.len().tolist()
df["html"] = df["html"].apply(prune_html)
pruned_sizes = df["html"].str.len().tolist()

prune_time = time.time() - prune_start

total_original = sum(original_sizes)
total_pruned = sum(pruned_sizes)
reduction_pct = (1 - total_pruned / max(total_original, 1)) * 100

print(f"  Pruning complete in {prune_time:.1f}s")
print(f"  Before: {total_original:>12,} chars total")
print(f"  After:  {total_pruned:>12,} chars total")
print(f"  Reduction: {reduction_pct:.1f}%")

# ---------------------------------------------------------------------------
# 3.4  Save newdataset-1.csv
# ---------------------------------------------------------------------------
# Ensure parent directory exists (OUTPUT_FILE_1 is an absolute path from config)
os.makedirs(os.path.dirname(OUTPUT_FILE_1), exist_ok=True)
print(f"\n  Saving {OUTPUT_FILE_1}...")
df.to_csv(OUTPUT_FILE_1, index=False, quoting=csv.QUOTE_ALL)
file_size = os.path.getsize(OUTPUT_FILE_1)
print(f"  Saved: {OUTPUT_FILE_1} ({file_size:,} bytes)")
print(f"  Columns: {list(df.columns)}")
print(f"  Rows: {len(df)}")

# ---------------------------------------------------------------------------
# 3.5  Dataset Statistics
# ---------------------------------------------------------------------------
print("\n  Dataset Statistics:")
print(f"    {'Metric':<30s} {'Value':>15s}")
print(f"    {'-'*30} {'-'*15}")
print(f"    {'Total rows':<30s} {len(df):>15,}")
print(f"    {'Avg HTML chars/row':<30s} {int(df['html'].str.len().mean()):>15,}")
print(f"    {'Min HTML chars/row':<30s} {int(df['html'].str.len().min()):>15,}")
print(f"    {'Max HTML chars/row':<30s} {int(df['html'].str.len().max()):>15,}")
print(f"    {'Empty HTML rows':<30s} {int((df['html'].str.len() == 0).sum()):>15,}")
print(f"    {'Unique page_ids':<30s} {df['page_id'].nunique():>15,}")

# Preview
print("\n  First 3 rows (truncated):")
for i, row in df.head(3).iterrows():
    html_preview = row['html'][:80] + "..." if len(row['html']) > 80 else row['html']
    print(f"    [{i}] page_id={row['page_id']}")
    print(f"        html={html_preview}")

print("\n  CELL 3 COMPLETE")
print("=" * 60)
