#!/usr/bin/env python3
"""
=============================================================================
  TPU CELL 3: DOWNLOAD DATASET & PRUNE HTML
=============================================================================
  - Downloads N rows from HuggingFace
  - Keeps ONLY page_id and html columns
  - Prunes invisible HTML tags (CPU-bound, parallelized)
  - Saves newdataset-1-tpu.csv
=============================================================================
"""

import os
import re
import csv
import sys
import time
from multiprocessing import Pool, cpu_count

import pandas as pd

# Setup path
for _p in [
    os.path.dirname(os.path.abspath(__file__)),
    os.path.join(os.getcwd(), "notebook_cells_tpu"),
    ".",
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tpu_config__1_ import ROW_COUNT, SOURCE_DATASET, OUTPUT_FILE_1, WORK_DIR

print("\n" + "=" * 60)
print("  TPU CELL 3: DOWNLOAD DATASET & PRUNE HTML")
print(f"  Rows: {ROW_COUNT} | Source: {SOURCE_DATASET}")
print("=" * 60)

# ---------------------------------------------------------------------------
# 3.1  Download
# ---------------------------------------------------------------------------
print("\n  Downloading...")
t0 = time.time()

from datasets import load_dataset
ds = load_dataset(SOURCE_DATASET, split=f"train[:{ROW_COUNT}]")
df = ds.to_pandas()

print(f"  Downloaded {len(df)} rows in {time.time()-t0:.1f}s")
print(f"  Raw columns: {list(df.columns)}")

# ---------------------------------------------------------------------------
# 3.2  Extract page_id + html only
# ---------------------------------------------------------------------------
col_map = {}
for col in df.columns:
    lo = col.lower().strip()
    if "page" in lo and "id" in lo:
        col_map["page_id"] = col
    if lo == "html":
        col_map["html"] = col

if "page_id" not in col_map or "html" not in col_map:
    col_map["page_id"] = df.columns[0]
    col_map["html"] = df.columns[1]

df = df[[col_map["page_id"], col_map["html"]]].copy()
df.columns = ["page_id", "html"]
print(f"  Columns: {list(df.columns)} | Rows: {len(df)}")

# ---------------------------------------------------------------------------
# 3.3  Prune HTML (parallelized across CPU cores)
# ---------------------------------------------------------------------------
print("  Pruning HTML...")
t0 = time.time()

from bs4 import BeautifulSoup, Comment

# Frozen sets for O(1) lookup
_INVISIBLE = frozenset([
    "style", "script", "noscript", "meta", "link",
    "iframe", "object", "embed", "svg", "path", "head",
])
_STRIP_ATTRS = frozenset([
    "class", "style", "onclick", "onload", "onmouseover",
    "onfocus", "onblur", "onchange", "onsubmit", "onerror",
    "onkeydown", "onkeyup", "onkeypress",
])
_SELF_CLOSING = frozenset(["br", "hr", "img", "input"])

# Pre-compiled regexes (avoid recompilation per row)
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def prune_html(raw: str) -> str:
    """Remove invisible elements, strip event handlers, collapse whitespace.

    Single-pass tag traversal (merged attribute stripping + empty removal)
    for better performance than the original two-pass approach.
    """
    if not raw or not isinstance(raw, str):
        return ""

    soup = BeautifulSoup(raw, "lxml")

    # Pass 1: Remove invisible tags and comments
    for tag in soup.find_all(_INVISIBLE):
        tag.decompose()
    for comment in soup.find_all(string=lambda x: isinstance(x, Comment)):
        comment.extract()

    # Pass 2 (merged): Strip attributes AND remove empty tags in one traversal
    # We iterate in reverse document order so child removals don't invalidate
    # the iterator for parent tags.
    for tag in reversed(soup.find_all(True)):
        # Strip dangerous/useless attributes
        to_del = [
            a for a in tag.attrs
            if a in _STRIP_ATTRS or a.startswith("data-")
        ]
        for a in to_del:
            del tag[a]

        # Remove empty non-self-closing tags
        if (
            tag.name not in _SELF_CLOSING
            and not tag.get_text(strip=True)
            and not tag.find_all(["img", "br", "hr", "input", "table"])
        ):
            tag.decompose()

    out = str(soup)
    out = _RE_MULTI_NEWLINE.sub("\n\n", out)
    out = _RE_MULTI_SPACE.sub(" ", out)
    return out.strip()


# Parallel pruning across CPU cores
orig_chars = df["html"].str.len().sum()

n_workers = min(cpu_count(), 8, len(df))  # cap at 8 workers
if n_workers > 1 and len(df) >= 4:
    print(f"  Using {n_workers} parallel workers...")
    with Pool(n_workers) as pool:
        df["html"] = pool.map(prune_html, df["html"].tolist())
else:
    df["html"] = df["html"].apply(prune_html)

pruned_chars = df["html"].str.len().sum()
reduction = (1 - pruned_chars / max(orig_chars, 1)) * 100
print(f"  Pruned in {time.time()-t0:.1f}s: {orig_chars:,} -> {pruned_chars:,} chars ({reduction:.1f}% reduction)")

# ---------------------------------------------------------------------------
# 3.4  Save
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_FILE_1) or ".", exist_ok=True)
df.to_csv(OUTPUT_FILE_1, index=False, quoting=csv.QUOTE_ALL)
print(f"  Saved: {OUTPUT_FILE_1} ({os.path.getsize(OUTPUT_FILE_1):,} bytes)")

lens = df["html"].str.len()
print(f"\n  Stats: avg={int(lens.mean())} chars/row, "
      f"min={int(lens.min())}, max={int(lens.max())}")

print("\n  TPU CELL 3 COMPLETE")
print("=" * 60)
