#!/usr/bin/env python3
"""
=============================================================================
  CELL 6: RUN MODEL ON ALL ROWS WITH TIME TRACKER
=============================================================================
  - Processes each row through the 3-task prompt
  - Displays live timing: layer time, row ETA, total ETA
  - Progress bar with speed metrics (tok/s, rows/hr)
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

try:
    from notebook_config import (
        ROW_COUNT, MODEL_ID, MAX_NEW_TOKENS, MAX_TOKENS_PER_ROW,
        HTML_BUDGET_RATIO, TOKENIZER_ID, SYSTEM_PROMPT,
        OUTPUT_FILE_1, CHECKPOINT_FILE, TEMPERATURE, TOP_P,
    )
except ImportError:
    sys.path.insert(0, os.path.join(os.getcwd(), "notebook_cells"))
    from notebook_config import (
        ROW_COUNT, MODEL_ID, MAX_NEW_TOKENS, MAX_TOKENS_PER_ROW,
        HTML_BUDGET_RATIO, TOKENIZER_ID, SYSTEM_PROMPT,
        OUTPUT_FILE_1, CHECKPOINT_FILE, TEMPERATURE, TOP_P,
    )

sys.path.insert(0, os.getcwd())
from timing_tracker import format_duration

print("\n" + "=" * 60)
print("  CELL 6: RUN INFERENCE ON ALL ROWS")
print(f"  Model: {MODEL_ID}")
print(f"  Rows: {ROW_COUNT}")
print("=" * 60)

# ---------------------------------------------------------------------------
# 6.1  Verify Prerequisites from Previous Cells
# ---------------------------------------------------------------------------
print("\n  Checking prerequisites...")

# These must exist from cell_5 (shared Colab/Kaggle kernel memory)
_missing = []
for _var in ["airllm_model", "model_tokenizer", "counting_tokenizer", "pipeline_timer"]:
    if _var not in dir():
        # Check globals (Colab cells share globals)
        try:
            eval(_var)
        except NameError:
            _missing.append(_var)

if _missing:
    print(f"  ERROR: Missing variables from Cell 5: {_missing}")
    print("  In Colab/Kaggle, you MUST run cells in order: 1 > 2 > 3 > 4 > 5 > 6")
    print("  Each cell runs in the same kernel, so variables persist between cells.")
    print("  If you restarted the runtime, re-run all cells from Cell 1.")
    raise RuntimeError(f"Run Cell 5 first. Missing: {_missing}")

print("  Model:           loaded")
print("  Tokenizer:       loaded")
print("  Pipeline timer:  active")

# Load the dataset from cell_3
if not os.path.exists(OUTPUT_FILE_1):
    print(f"  ERROR: {OUTPUT_FILE_1} not found. Run Cell 3 first.")
    raise FileNotFoundError(f"Run Cell 3 first: {OUTPUT_FILE_1}")

df = pd.read_csv(OUTPUT_FILE_1)
# Fill NaN values to prevent str operation failures on empty HTML rows
df["html"] = df["html"].fillna("")
df["page_id"] = df["page_id"].fillna("")
print(f"  Dataset:         {len(df)} rows loaded from {OUTPUT_FILE_1}")

# ---------------------------------------------------------------------------
# 6.2  Helper Functions
# ---------------------------------------------------------------------------
def count_tokens(text: str) -> int:
    """Count tokens using the counting tokenizer."""
    if not text:
        return 0
    return len(counting_tokenizer.encode(text, add_special_tokens=False))


def truncate_text(text: str, max_tokens: int) -> str:
    """Truncate text to fit within max_tokens."""
    ids = counting_tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return counting_tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def build_prompt(html_content: str) -> str:
    """Build the chat prompt."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": html_content},
    ]
    if hasattr(model_tokenizer, "apply_chat_template"):
        return model_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{html_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def validate_response(response: str) -> Dict:
    """Check if response contains all 3 tasks."""
    if not response:
        return {"valid": False, "tasks_found": 0}
    markers = ["===TASK 1===", "===TASK 2===", "===TASK 3==="]
    found = sum(1 for m in markers if m in response)
    has_query = "USER_QUERY:" in response
    return {
        "valid": found == 3 and has_query,
        "tasks_found": found,
    }


def generate_response(html_content: str) -> Tuple[Optional[str], int]:
    """Generate model response for one HTML row."""
    import torch

    prompt = build_prompt(html_content)
    input_ids = model_tokenizer(prompt, return_tensors="pt").input_ids

    # Reset layer timer
    pipeline_timer.layer_timer.reset_for_new_generation()

    try:
        gen_start = time.time()

        output = airllm_model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            return_dict_in_generate=True,
        )

        gen_time = time.time() - gen_start

        # Decode new tokens only
        new_ids = output.sequences[0][input_ids.shape[1]:]
        token_count = len(new_ids)
        response = model_tokenizer.decode(new_ids, skip_special_tokens=True)

        pipeline_timer.layer_timer.complete_pass()

        tok_per_sec = token_count / gen_time if gen_time > 0 else 0
        print(f"    Generated {token_count} tokens in {format_duration(gen_time)} ({tok_per_sec:.1f} tok/s)")

        # Cleanup
        del output, new_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response.strip(), token_count

    except Exception as e:
        print(f"    ERROR: {e}")
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return None, 0


# ---------------------------------------------------------------------------
# 6.3  Load Checkpoint (if resuming)
# ---------------------------------------------------------------------------
checkpoint_path = Path(CHECKPOINT_FILE)
completed_indices = []
responses = {}

if checkpoint_path.exists():
    try:
        ckpt = json.loads(checkpoint_path.read_text())
        completed_indices = ckpt.get("completed", [])
        responses = {int(k): v for k, v in ckpt.get("responses", {}).items()}
        print(f"\n  Checkpoint found: {len(completed_indices)} rows already done")
    except Exception:
        print("\n  Checkpoint corrupted, starting fresh")

# ---------------------------------------------------------------------------
# 6.4  Pre-truncate Oversized HTML
# ---------------------------------------------------------------------------
print("\n  Pre-truncating oversized HTML...")
html_budget = int(MAX_TOKENS_PER_ROW * HTML_BUDGET_RATIO)
truncated_count = 0
for idx in df.index:
    if count_tokens(df.at[idx, "html"]) > html_budget:
        df.at[idx, "html"] = truncate_text(df.at[idx, "html"], html_budget)
        truncated_count += 1
print(f"  Truncated {truncated_count} rows to fit {html_budget} token budget")

# ---------------------------------------------------------------------------
# 6.5  MAIN INFERENCE LOOP
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  STARTING INFERENCE")
print("=" * 60)

pipeline_timer.start_phase("generation")
pipeline_timer.row_timer.start_generation_phase()

stats = {
    "success": 0,
    "failed": 0,
    "skipped": len(completed_indices),
    "valid": 0,
    "total_tokens": 0,
}

for idx, row in df.iterrows():
    # Skip if already done (checkpoint)
    if idx in completed_indices:
        print(f"  Row {idx+1}/{len(df)} - skipped (checkpoint)")
        pipeline_timer.row_timer.skip_row(idx)
        continue

    # Start row
    print(f"\n  {'='*55}")
    print(f"  Row {idx+1}/{len(df)} | page_id={row['page_id']} | html={len(row['html'])} chars")
    print(f"  {'='*55}")

    pipeline_timer.row_timer.start_row(idx)

    # Generate
    response, output_tokens = generate_response(row["html"])

    if response is None:
        response = "[ERROR] Generation failed."
        stats["failed"] += 1
    else:
        stats["success"] += 1
        stats["total_tokens"] += output_tokens

        # Validate
        val = validate_response(response)
        if val["valid"]:
            stats["valid"] += 1
            print(f"    Validation: PASS (3/3 tasks)")
        else:
            print(f"    Validation: PARTIAL ({val['tasks_found']}/3 tasks)")

    # End row timer
    pipeline_timer.row_timer.end_row(idx, output_tokens=output_tokens)

    # Store
    responses[idx] = response
    completed_indices.append(idx)

    # Checkpoint every 5 rows
    if len(completed_indices) % 5 == 0:
        ckpt_data = {
            "completed": completed_indices,
            "responses": {str(k): v for k, v in responses.items()},
            "timestamp": time.time(),
        }
        checkpoint_path.write_text(json.dumps(ckpt_data, ensure_ascii=False))
        print(f"    Checkpoint saved ({len(completed_indices)} rows)")

pipeline_timer.end_phase("generation")

# ---------------------------------------------------------------------------
# 6.6  Inference Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  INFERENCE COMPLETE")
print("=" * 60)
print(f"  Total rows:     {len(df)}")
print(f"  Successful:     {stats['success']}")
print(f"  Valid (3/3):    {stats['valid']}")
print(f"  Failed:         {stats['failed']}")
print(f"  Skipped:        {stats['skipped']}")
print(f"  Total tokens:   {stats['total_tokens']:,}")

if stats['success'] > 0:
    avg_tokens = stats['total_tokens'] / stats['success']
    print(f"  Avg tokens/row: {avg_tokens:.0f}")

print("\n  CELL 6 COMPLETE - Responses generated")
print("  Run Cell 7 to save final CSVs and timing report")
print("=" * 60)

# These variables carry forward to cell_7:
# df, responses, pipeline_timer, counting_tokenizer, stats
