#!/usr/bin/env python3
"""
=============================================================================
  TPU CELL 3: DOWNLOAD DATASET & PRUNE HTML
=============================================================================
  Identical logic to GPU version - dataset processing is CPU-bound.
  - Downloads N rows from HuggingFace
  - Keeps ONLY page_id and html columns
  - Prunes invisible HTML tags
  - Saves newdataset-1-tpu.csv
=============================================================================
"""

import os
import re
import csv
import sys
import time

import pandas as pd

for _p in ["notebook_cells_tpu", os.path.join(os.getcwd(), "notebook_cells_tpu"), "."]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tpu_config import ROW_COUNT, SOURCE_DATASET, OUTPUT_FILE_1, WORK_DIR

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
# 3.3  Prune HTML
# ---------------------------------------------------------------------------
print("  Pruning HTML...")
t0 = time.time()

from bs4 import BeautifulSoup, Comment

INVISIBLE = frozenset(["style","script","noscript","meta","link","iframe","object","embed","svg","path","head"])
STRIP = frozenset(["class","style","onclick","onload","onmouseover","onfocus","onblur","onchange","onsubmit","onerror","onkeydown","onkeyup","onkeypress"])
SELF_CLOSING = frozenset(["br","hr","img","input"])

def prune_html(raw):
    if not raw or not isinstance(raw, str):
        return ""
    soup = BeautifulSoup(raw, "lxml")
    for t in INVISIBLE:
        for tag in soup.find_all(t): tag.decompose()
    for c in soup.find_all(string=lambda x: isinstance(x, Comment)): c.extract()
    for tag in soup.find_all(True):
        for a in [a for a in tag.attrs if a in STRIP or a.startswith("data-")]: del tag[a]
    for tag in soup.find_all(True):
        if tag.name not in SELF_CLOSING and not tag.get_text(strip=True) and not tag.find_all(["img","br","hr","input","table"]):
            tag.decompose()
    out = str(soup)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()

orig = df["html"].str.len().sum()
df["html"] = df["html"].apply(prune_html)
pruned = df["html"].str.len().sum()
print(f"  Pruned in {time.time()-t0:.1f}s: {orig:,} -> {pruned:,} chars ({(1-pruned/max(orig,1))*100:.1f}% reduction)")

# ---------------------------------------------------------------------------
# 3.4  Save
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_FILE_1), exist_ok=True)
df.to_csv(OUTPUT_FILE_1, index=False, quoting=csv.QUOTE_ALL)
print(f"  Saved: {OUTPUT_FILE_1} ({os.path.getsize(OUTPUT_FILE_1):,} bytes)")

print(f"\n  Stats: avg={int(df['html'].str.len().mean())} chars/row, "
      f"min={int(df['html'].str.len().min())}, max={int(df['html'].str.len().max())}")

print("\n  TPU CELL 3 COMPLETE")
print("=" * 60)
