#!/usr/bin/env python3
"""
=============================================================================
  TPU CELL 5: LOAD MODEL ONTO ACCELERATOR
=============================================================================
  - Downloads model weights from HuggingFace
  - Loads model in bfloat16 (TPU-native) or float16 (GPU)
  - Moves model to the best available device (TPU > GPU > CPU)
  - Loads tokenizers (model tokenizer + counting tokenizer)
  - Initializes PipelineTimer
  - Runs verification inference
=============================================================================
"""

import os
import sys
import gc
import time
from pathlib import Path

for _p in [
    os.path.dirname(os.path.abspath(__file__)),
    os.path.join(os.getcwd(), "notebook_cells_tpu"),
    ".",
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tpu_config__1_ import (
    MODEL_ID, DTYPE, CACHE_DIR, TOKENIZER_ID,
    ROW_COUNT, MAX_NEW_TOKENS, SYSTEM_PROMPT,
    TIMING_REPORT_FILE,
)
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
# 5.2  Resolve Device & Dtype
# ---------------------------------------------------------------------------
import torch

# Determine best device
device = None
device_type = "cpu"

try:
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    _t = torch.tensor([1.0], device=device)
    _ = _t.cpu()
    device_type = "tpu"
    print(f"\n  Device: TPU ({device})")
except Exception:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "gpu"
        print(f"\n  Device: GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        device_type = "cpu"
        print(f"\n  Device: CPU")

# Resolve dtype
dtype_map = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
torch_dtype = dtype_map.get(DTYPE, torch.bfloat16)

# GPU doesn't always support bfloat16 well; prefer float16
if device_type == "gpu" and torch_dtype == torch.bfloat16:
    if not torch.cuda.is_bf16_supported():
        torch_dtype = torch.float16
        print("  Note: GPU doesn't support bfloat16, using float16")

print(f"  Torch dtype: {torch_dtype}")

# ---------------------------------------------------------------------------
# 5.3  Load Model
# ---------------------------------------------------------------------------
print(f"\n  Loading model: {MODEL_ID}")
print("  Step 1: Downloading weights (first run downloads from HF)...")

model_load_start = time.time()

from transformers import AutoModelForCausalLM, AutoTokenizer

load_kwargs = {
    "torch_dtype": torch_dtype,
    "cache_dir": CACHE_DIR,
    "trust_remote_code": True,
    "low_cpu_mem_usage": True,
}

# For GPU, use device_map="auto" for automatic sharding
if device_type == "gpu":
    load_kwargs["device_map"] = "auto"

tpu_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)

cpu_load_time = time.time() - model_load_start
print(f"  Weights loaded in {format_duration(cpu_load_time)}")

# Move to device (TPU needs explicit .to(); GPU with device_map is already placed)
if device_type == "tpu":
    print("  Step 2: Moving model to TPU...")
    move_start = time.time()
    tpu_model = tpu_model.to(device)
    move_time = time.time() - move_start
    print(f"  TPU transfer: {format_duration(move_time)}")
elif device_type == "cpu":
    print("  Step 2: Model stays on CPU")
else:
    print("  Step 2: Model placed on GPU via device_map='auto'")

tpu_model.eval()
total_load_time = time.time() - model_load_start

# Count parameters
total_params = sum(p.numel() for p in tpu_model.parameters())
print(f"  Total load:  {format_duration(total_load_time)}")
print(f"  Parameters:  {total_params/1e9:.2f}B")

# ---------------------------------------------------------------------------
# 5.4  Load Tokenizers
# ---------------------------------------------------------------------------
print(f"\n  Loading tokenizer: {MODEL_ID}")
tok_start = time.time()

model_tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, trust_remote_code=True, cache_dir=CACHE_DIR,
)

# Counting tokenizer (lightweight, same vocab as model family)
if TOKENIZER_ID != MODEL_ID:
    print(f"  Loading counting tokenizer: {TOKENIZER_ID}")
    counting_tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID, trust_remote_code=True, cache_dir=CACHE_DIR,
    )
else:
    counting_tokenizer = model_tokenizer

print(f"  Tokenizers loaded in {format_duration(time.time()-tok_start)}")

pipeline_timer.end_phase("model_loading")

# ---------------------------------------------------------------------------
# 5.5  Verification Inference
# ---------------------------------------------------------------------------
print("\n  Running verification inference...")
verify_start = time.time()

try:
    test_messages = [
        {"role": "system", "content": "Reply with exactly: DEVICE_OK"},
        {"role": "user", "content": "<p>test</p>"},
    ]

    if hasattr(model_tokenizer, "apply_chat_template"):
        prompt = model_tokenizer.apply_chat_template(
            test_messages, tokenize=False, add_generation_prompt=True,
        )
    else:
        prompt = (
            "<|im_start|>system\nReply with exactly: DEVICE_OK<|im_end|>\n"
            "<|im_start|>user\n<p>test</p><|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    input_ids = model_tokenizer(prompt, return_tensors="pt").input_ids
    if device_type == "tpu":
        input_ids = input_ids.to(device)
    elif device_type == "gpu":
        input_ids = input_ids.to("cuda")

    with torch.no_grad():
        output = tpu_model.generate(
            input_ids, max_new_tokens=10, do_sample=False,
        )

    # XLA mark_step to flush the computation graph
    if device_type == "tpu":
        import torch_xla.core.xla_model as xm
        xm.mark_step()

    result = model_tokenizer.decode(
        output[0][input_ids.shape[1]:], skip_special_tokens=True,
    )
    verify_time = time.time() - verify_start

    print(f"  Output:  '{result.strip()[:60]}'")
    print(f"  Time:    {format_duration(verify_time)}")
    print(f"  Status:  MODEL READY ON {device_type.upper()}")

    del output, input_ids
    gc.collect()

except Exception as e:
    print(f"  Verification failed: {e}")
    print("  Model may still work for longer sequences.")

# ---------------------------------------------------------------------------
# 5.6  Summary
# ---------------------------------------------------------------------------
# Store device info for Cell 6
tpu_device = device

print("\n" + "-" * 60)
print(f"  MODEL READY ON {device_type.upper()}")
print(f"  Model:      {MODEL_ID}")
print(f"  Params:     {total_params/1e9:.2f}B")
print(f"  Dtype:      {torch_dtype}")
print(f"  Device:     {device}")
print(f"  Layers:     {num_layers}")
print(f"  Load time:  {format_duration(total_load_time)}")
print(f"  Chat tmpl:  {'yes' if hasattr(model_tokenizer, 'apply_chat_template') else 'manual'}")

print("\n  TPU CELL 5 COMPLETE")
print("=" * 60)

# Carry forward: tpu_model, model_tokenizer, counting_tokenizer,
#                pipeline_timer, tpu_device, device_type
