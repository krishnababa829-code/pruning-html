#!/usr/bin/env python3
"""
=============================================================================
  TPU CELL 6: RUN INFERENCE ON ALL ROWS
=============================================================================
  - Processes each row through the 3-task prompt on TPU
  - Uses torch.no_grad() + XLA mark_step() for optimal TPU execution
  - Live timing: row ETA, total ETA, tokens/sec
  - Checkpoint/resume for interrupted sessions
  - Validates each response has all 3 tasks
=============================================================================
"""

import os
import re
import gc
import sys
import csv
import json
import time
from pathlib import Path
from typing import Optional, Dict, Tuple

import pandas as pd
import torch

for _p in ["notebook_cells_tpu", os.path.join(os.getcwd(), "notebook_cells_tpu"), "."]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tpu_config import (
    ROW_COUNT, MODEL_ID, MAX_NEW_TOKENS, MAX_TOKENS_PER_ROW,
    HTML_BUDGET_RATIO, SYSTEM_PROMPT, OUTPUT_FILE_1, CHECKPOINT_FILE,
)

for _p in [os.getcwd(), os.path.dirname(os.getcwd())]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from timing_tracker import format_duration

print("\n" + "=" * 60)
print("  TPU CELL 6: RUN INFERENCE")
print(f"  Model: {MODEL_ID} | Rows: {ROW_COUNT}")
print("=" * 60)

# ---------------------------------------------------------------------------
# 6.1  Prerequisites
# ---------------------------------------------------------------------------
_missing = []
for _var in ["tpu_model", "model_tokenizer", "counting_tokenizer", "pipeline_timer", "tpu_device"]:
    try:
        eval(_var)
    except NameError:
        _missing.append(_var)

if _missing:
    print(f"  ERROR: Missing from Cell 5: {_missing}")
    print("  Run cells in order: 1 > 2 > 3 > 4 > 5 > 6")
    raise RuntimeError(f"Run Cell 5 first. Missing: {_missing}")

print("  Model:     loaded on TPU")
print("  Tokenizer: loaded")

if not os.path.exists(OUTPUT_FILE_1):
    raise FileNotFoundError(f"Run Cell 3 first: {OUTPUT_FILE_1}")

df = pd.read_csv(OUTPUT_FILE_1)
df["html"] = df["html"].fillna("")
df["page_id"] = df["page_id"].fillna("")
print(f"  Dataset:   {len(df)} rows")

# ---------------------------------------------------------------------------
# 6.2  Helpers
# ---------------------------------------------------------------------------
def count_tokens(text):
    if not text: return 0
    return len(counting_tokenizer.encode(text, add_special_tokens=False))

def truncate_text(text, max_tok):
    ids = counting_tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tok: return text
    return counting_tokenizer.decode(ids[:max_tok], skip_special_tokens=True)

def build_prompt(html):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": html}]
    if hasattr(model_tokenizer, "apply_chat_template"):
        return model_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{html}<|im_end|>\n<|im_start|>assistant\n"

def validate_response(resp):
    if not resp: return {"valid": False, "tasks_found": 0}
    found = sum(1 for m in ["===TASK 1===", "===TASK 2===", "===TASK 3==="] if m in resp)
    return {"valid": found == 3 and "USER_QUERY:" in resp, "tasks_found": found}

def generate_on_tpu(html):
    """Generate response using TPU with XLA optimization."""
    prompt = build_prompt(html)
    input_ids = model_tokenizer(prompt, return_tensors="pt").input_ids.to(tpu_device)

    try:
        gen_start = time.time()

        with torch.no_grad():
            output = tpu_model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
            )

        gen_time = time.time() - gen_start

        new_ids = output[0][input_ids.shape[1]:]
        token_count = len(new_ids)
        response = model_tokenizer.decode(new_ids, skip_special_tokens=True)

        tok_s = token_count / gen_time if gen_time > 0 else 0
        print(f"    {token_count} tokens in {format_duration(gen_time)} ({tok_s:.1f} tok/s)")

        del output, new_ids, input_ids
        gc.collect()

        return response.strip(), token_count

    except Exception as e:
        print(f"    ERROR: {e}")
        gc.collect()
        return None, 0

# ---------------------------------------------------------------------------
# 6.3  Checkpoint
# ---------------------------------------------------------------------------
ckpt_path = Path(CHECKPOINT_FILE)
completed_indices = []
responses = {}

if ckpt_path.exists():
    try:
        ckpt = json.loads(ckpt_path.read_text())
        completed_indices = ckpt.get("completed", [])
        responses = {int(k): v for k, v in ckpt.get("responses", {}).items()}
        print(f"  Checkpoint: {len(completed_indices)} rows done")
    except Exception:
        print("  Checkpoint corrupted, starting fresh")

# ---------------------------------------------------------------------------
# 6.4  Pre-truncate
# ---------------------------------------------------------------------------
html_budget = int(MAX_TOKENS_PER_ROW * HTML_BUDGET_RATIO)
trunc = 0
for idx in df.index:
    if count_tokens(df.at[idx, "html"]) > html_budget:
        df.at[idx, "html"] = truncate_text(df.at[idx, "html"], html_budget)
        trunc += 1
print(f"  Pre-truncated: {trunc} rows")

# ---------------------------------------------------------------------------
# 6.5  INFERENCE LOOP
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  STARTING TPU INFERENCE")
print("=" * 60)

pipeline_timer.start_phase("generation")
pipeline_timer.row_timer.start_generation_phase()

stats = {"success": 0, "failed": 0, "skipped": len(completed_indices), "valid": 0, "total_tokens": 0}

for idx, row in df.iterrows():
    if idx in completed_indices:
        print(f"  Row {idx+1}/{len(df)} - skipped")
        pipeline_timer.row_timer.skip_row(idx)
        continue

    print(f"\n  Row {idx+1}/{len(df)} | page_id={row['page_id']} | {len(row['html'])} chars")
    pipeline_timer.row_timer.start_row(idx)

    response, out_tok = generate_on_tpu(row["html"])

    if response is None:
        response = "[ERROR] Generation failed."
        stats["failed"] += 1
    else:
        stats["success"] += 1
        stats["total_tokens"] += out_tok
        val = validate_response(response)
        if val["valid"]:
            stats["valid"] += 1
            print(f"    Validation: PASS (3/3)")
        else:
            print(f"    Validation: {val['tasks_found']}/3 tasks")

    pipeline_timer.row_timer.end_row(idx, output_tokens=out_tok)
    responses[idx] = response
    completed_indices.append(idx)

    if len(completed_indices) % 5 == 0:
        ckpt_path.write_text(json.dumps({
            "completed": completed_indices,
            "responses": {str(k): v for k, v in responses.items()},
            "timestamp": time.time(),
        }, ensure_ascii=False))
        print(f"    Checkpoint saved ({len(completed_indices)} rows)")

pipeline_timer.end_phase("generation")

print("\n" + "=" * 60)
print("  TPU INFERENCE COMPLETE")
print(f"  Success: {stats['success']} | Valid: {stats['valid']} | Failed: {stats['failed']}")
print(f"  Tokens: {stats['total_tokens']:,}")
print("\n  TPU CELL 6 COMPLETE")
print("=" * 60)
