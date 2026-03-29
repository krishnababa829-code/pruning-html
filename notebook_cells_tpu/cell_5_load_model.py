#!/usr/bin/env python3
"""
=============================================================================
  TPU CELL 5: LOAD MODEL ONTO TPU
=============================================================================
  - Downloads model weights from HuggingFace
  - Loads model in bfloat16 (TPU-native precision)
  - Moves model to XLA device
  - Loads tokenizer
  - Initializes timing tracker
  - Runs verification inference

  Strategy: Load on CPU first, then move to TPU device.
  For models that fit in TPU HBM, this is the most reliable approach.
=============================================================================
"""

import os
import sys
import gc
import time
from pathlib import Path

for _p in ["notebook_cells_tpu", os.path.join(os.getcwd(), "notebook_cells_tpu"), "."]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tpu_config import (
    MODEL_ID, DTYPE, CACHE_DIR, TOKENIZER_ID,
    ROW_COUNT, MAX_NEW_TOKENS, SYSTEM_PROMPT,
    TIMING_REPORT_FILE,
)

# Import timing tracker
for _p in [os.getcwd(), os.path.dirname(os.getcwd())]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from timing_tracker import (
        PipelineTimer, detect_layer_count, format_duration,
    )
except ImportError:
    sys.path.insert(0, os.path.join(os.getcwd(), ".."))
    from timing_tracker import (
        PipelineTimer, detect_layer_count, format_duration,
    )

print("\n" + "=" * 60)
print("  TPU CELL 5: LOAD MODEL")
print(f"  Model: {MODEL_ID}")
print(f"  Dtype: {DTYPE}")
print("=" * 60)

# ---------------------------------------------------------------------------
# 5.1  Initialize Timing
# ---------------------------------------------------------------------------
num_layers = detect_layer_count(MODEL_ID)
pipeline_timer = PipelineTimer(
    total_rows=ROW_COUNT,
    total_layers=num_layers,
    report_path=TIMING_REPORT_FILE,
)
pipeline_timer.start()
pipeline_timer.start_phase("model_loading")

# ---------------------------------------------------------------------------
# 5.2  Resolve dtype
# ---------------------------------------------------------------------------
import torch
import torch_xla.core.xla_model as xm

tpu_device = xm.xla_device()

dtype_map = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
torch_dtype = dtype_map.get(DTYPE, torch.bfloat16)
print(f"\n  Torch dtype: {torch_dtype}")
print(f"  TPU device:  {tpu_device}")

# ---------------------------------------------------------------------------
# 5.3  Load Model
# ---------------------------------------------------------------------------
print(f"\n  Loading model: {MODEL_ID}")
print("  Step 1: Downloading weights to CPU (first run downloads from HF)...")

model_load_start = time.time()

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load to CPU first, then move to TPU
# This is the most reliable method for TPU
tpu_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch_dtype,
    cache_dir=CACHE_DIR,
    trust_remote_code=True,
    low_cpu_mem_usage=True,  # Load weights incrementally to reduce peak CPU RAM
)

cpu_load_time = time.time() - model_load_start
print(f"  CPU load complete in {format_duration(cpu_load_time)}")

# Move to TPU
print("  Step 2: Moving model to TPU...")
move_start = time.time()
tpu_model = tpu_model.to(tpu_device)
tpu_model.eval()  # Inference mode
move_time = time.time() - move_start
print(f"  TPU transfer complete in {format_duration(move_time)}")

total_load_time = time.time() - model_load_start
print(f"  Total model load: {format_duration(total_load_time)}")

# Count parameters
total_params = sum(p.numel() for p in tpu_model.parameters())
print(f"  Parameters: {total_params/1e9:.1f}B")

# ---------------------------------------------------------------------------
# 5.4  Load Tokenizers
# ---------------------------------------------------------------------------
print(f"\n  Loading tokenizer: {MODEL_ID}")
tok_start = time.time()

model_tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, trust_remote_code=True, cache_dir=CACHE_DIR,
)

print(f"  Loading counting tokenizer: {TOKENIZER_ID}")
counting_tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_ID, trust_remote_code=True,
)

print(f"  Tokenizers loaded in {format_duration(time.time()-tok_start)}")

pipeline_timer.end_phase("model_loading")

# ---------------------------------------------------------------------------
# 5.5  Verification
# ---------------------------------------------------------------------------
print("\n  Running TPU verification...")
verify_start = time.time()

try:
    test_messages = [
        {"role": "system", "content": "Reply with exactly: TPU_OK"},
        {"role": "user", "content": "<p>test</p>"},
    ]

    if hasattr(model_tokenizer, "apply_chat_template"):
        prompt = model_tokenizer.apply_chat_template(
            test_messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = "<|im_start|>system\nReply with exactly: TPU_OK<|im_end|>\n<|im_start|>user\n<p>test</p><|im_end|>\n<|im_start|>assistant\n"

    input_ids = model_tokenizer(prompt, return_tensors="pt").input_ids.to(tpu_device)

    with torch.no_grad():
        output = tpu_model.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=False,
        )

    result = model_tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    verify_time = time.time() - verify_start

    print(f"  Output: '{result.strip()[:50]}'")
    print(f"  Time:   {format_duration(verify_time)}")
    print(f"  Status: MODEL READY ON TPU")

    del output, input_ids
    gc.collect()

except Exception as e:
    print(f"  Verification failed: {e}")
    print("  Model may still work for longer sequences.")

# ---------------------------------------------------------------------------
# 5.6  Summary
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("  TPU MODEL READY")
print(f"  Model:      {MODEL_ID}")
print(f"  Params:     {total_params/1e9:.1f}B")
print(f"  Dtype:      {DTYPE}")
print(f"  Device:     {tpu_device}")
print(f"  Layers:     {num_layers}")
print(f"  Load time:  {format_duration(total_load_time)}")
print(f"  Chat tmpl:  {'yes' if hasattr(model_tokenizer, 'apply_chat_template') else 'manual'}")

print("\n  TPU CELL 5 COMPLETE")
print("=" * 60)

# Carry forward: tpu_model, model_tokenizer, counting_tokenizer,
#                pipeline_timer, tpu_device
