#!/usr/bin/env python3
"""
=============================================================================
  CELL 5: PREPARE & LOAD THE QWEN MODEL VIA AirLLM
=============================================================================
  - Downloads model weights (layer by layer)
  - Initializes AirLLM with compression
  - Loads tokenizer for prompt formatting
  - Initializes the timing tracker system
  - Verifies model is ready for inference
=============================================================================
"""

import os
import sys
import gc
import time
import json
from pathlib import Path

try:
    from notebook_config import (
        MODEL_ID, COMPRESSION, CACHE_DIR, TOKENIZER_ID,
        ROW_COUNT, MAX_TOKENS_PER_ROW, MAX_NEW_TOKENS,
        HTML_BUDGET_RATIO, SYSTEM_PROMPT,
    )
except ImportError:
    sys.path.insert(0, os.path.join(os.getcwd(), "notebook_cells"))
    from notebook_config import (
        MODEL_ID, COMPRESSION, CACHE_DIR, TOKENIZER_ID,
        ROW_COUNT, MAX_TOKENS_PER_ROW, MAX_NEW_TOKENS,
        HTML_BUDGET_RATIO, SYSTEM_PROMPT,
    )

# Import timing tracker from the repo
# Handle multiple possible locations (Colab cwd may differ)
for _p in [os.getcwd(), os.path.join(os.getcwd(), ".."), "notebook_cells", "."]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from timing_tracker import (
        LayerTimer, RowTimer, PipelineTimer,
        install_layer_hooks, detect_layer_count, format_duration,
    )
except ImportError:
    # If cwd is notebook_cells/, go up one level
    sys.path.insert(0, os.path.dirname(os.getcwd()))
    from timing_tracker import (
        LayerTimer, RowTimer, PipelineTimer,
        install_layer_hooks, detect_layer_count, format_duration,
    )

print("\n" + "=" * 60)
print("  CELL 5: PREPARE & LOAD MODEL")
print(f"  Model: {MODEL_ID}")
print(f"  Compression: {COMPRESSION}")
print("=" * 60)

# ---------------------------------------------------------------------------
# 5.1  Initialize Timing System
# ---------------------------------------------------------------------------
print("\n  Initializing timing tracker...")

num_layers = detect_layer_count(MODEL_ID)

pipeline_timer = PipelineTimer(
    total_rows=ROW_COUNT,
    total_layers=num_layers,
    report_path="timing_report.json",
)
pipeline_timer.start()
pipeline_timer.start_phase("model_loading")

print(f"  Model layers: {num_layers}")
print(f"  Total rows to process: {ROW_COUNT}")

# ---------------------------------------------------------------------------
# 5.2  Load AirLLM Model
# ---------------------------------------------------------------------------
print(f"\n  Loading AirLLM model: {MODEL_ID}")
print("  This downloads model weights layer-by-layer (first run takes time)...")

model_load_start = time.time()

from airllm import AutoModel as AirAutoModel

model_kwargs = {
    "cache_dir": CACHE_DIR,
    "trust_remote_code": True,
}

if COMPRESSION in ("4bit", "8bit"):
    model_kwargs["compression"] = COMPRESSION

airllm_model = AirAutoModel.from_pretrained(MODEL_ID, **model_kwargs)

model_load_time = time.time() - model_load_start
print(f"  Model loaded in {format_duration(model_load_time)}")

# Install layer timing hooks
print("\n  Installing layer timing hooks...")
hooks_installed = install_layer_hooks(airllm_model, pipeline_timer.layer_timer)

# ---------------------------------------------------------------------------
# 5.3  Load Tokenizer
# ---------------------------------------------------------------------------
print(f"\n  Loading tokenizer: {MODEL_ID}")
tokenizer_start = time.time()

from transformers import AutoTokenizer

model_tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    cache_dir=CACHE_DIR,
)

# Also load the counting tokenizer (smaller, faster)
print(f"  Loading counting tokenizer: {TOKENIZER_ID}")
counting_tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_ID,
    trust_remote_code=True,
)

tokenizer_time = time.time() - tokenizer_start
print(f"  Tokenizers loaded in {format_duration(tokenizer_time)}")

pipeline_timer.end_phase("model_loading")

# ---------------------------------------------------------------------------
# 5.4  Verify Model with a Quick Test
# ---------------------------------------------------------------------------
print("\n  Running quick verification...")
verify_start = time.time()

try:
    import torch

    test_prompt = "<p>Hello World</p>"
    messages = [
        {"role": "system", "content": "Reply with: TEST OK"},
        {"role": "user", "content": test_prompt},
    ]

    if hasattr(model_tokenizer, "apply_chat_template"):
        formatted = model_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted = f"<|im_start|>system\nReply with: TEST OK<|im_end|>\n<|im_start|>user\n{test_prompt}<|im_end|>\n<|im_start|>assistant\n"

    input_ids = model_tokenizer(formatted, return_tensors="pt").input_ids

    output = airllm_model.generate(
        input_ids,
        max_new_tokens=20,
        use_cache=True,
        return_dict_in_generate=True,
    )

    test_output = model_tokenizer.decode(
        output.sequences[0][input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    verify_time = time.time() - verify_start
    print(f"  Verification output: '{test_output.strip()[:50]}'")
    print(f"  Verification time: {format_duration(verify_time)}")
    print("  Model is READY for inference")

    # Cleanup
    del output, input_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

except Exception as e:
    print(f"  Verification failed: {e}")
    print("  Model may still work - proceeding...")

# ---------------------------------------------------------------------------
# 5.5  Summary
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("  MODEL READY")
print(f"  Model:          {MODEL_ID}")
print(f"  Layers:         {num_layers}")
print(f"  Compression:    {COMPRESSION}")
print(f"  Load time:      {format_duration(model_load_time)}")
print(f"  Layer hooks:    {'installed' if hooks_installed else 'row-level timing only'}")
print(f"  Chat template:  {'yes' if hasattr(model_tokenizer, 'apply_chat_template') else 'manual'}")

try:
    import torch
    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
        print(f"  VRAM remaining: {free_mem:.1f} GB")
except Exception:
    pass

print("\n  CELL 5 COMPLETE")
print("=" * 60)

# These variables are used by cell_6:
# airllm_model, model_tokenizer, counting_tokenizer, pipeline_timer
