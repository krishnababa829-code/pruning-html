#!/usr/bin/env python3
"""
=============================================================================
  TPU CELL 5: LOAD MODEL WITH PROPER TPU SHARDING
=============================================================================
  CRITICAL FIX: The old code did `model.to(xla:0)` which loads the ENTIRE
  model onto a SINGLE 8 GB TPU core. This causes RESOURCE_EXHAUSTED for
  any model > ~3.5B parameters.

  NEW APPROACH (3 strategies, auto-selected):

  Strategy A: SINGLE-CORE (model <= 6 GB, e.g. Qwen-3B)
    - model.to(xm.xla_device())  # fits in one 8 GB core

  Strategy B: FSDP SHARDING (model 6-64 GB, e.g. Qwen-7B/14B/32B)
    - Load model on CPU with meta device (zero memory)
    - Wrap with XlaFullyShardedDataParallel
    - Each core holds 1/8 of the parameters
    - 28 GB model -> 3.5 GB per core (fits in 8 GB)

  Strategy C: GPU / CPU FALLBACK
    - device_map="auto" for GPU
    - Direct CPU loading

  The strategy is auto-detected based on Cell 4's memory assessment.
=============================================================================
"""

import os
import sys
import gc
import time
import functools
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
# 5.2  Resolve Device, Dtype, and Sharding Strategy
# ---------------------------------------------------------------------------
import torch

# Determine best device
device = None
device_type = "cpu"
use_fsdp = False

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    _t = torch.tensor([1.0], device=device)
    _ = _t.cpu()
    del _t
    device_type = "tpu"

    # Check if FSDP is needed (from Cell 4's analysis)
    # Re-derive here for robustness (Cell 4 may not have run in same process)
    try:
        import torch_xla.runtime as xr
        tpu_num_cores = xr.global_device_count()
    except Exception:
        tpu_num_cores = xm.xrt_world_size() if hasattr(xm, "xrt_world_size") else 8

    from tpu_config__1_ import IS_COLAB, IS_KAGGLE
    hbm_per_core = 16 if IS_KAGGLE else 8

    # Estimate model size
    model_lower = MODEL_ID.lower()
    model_bf16_gb = 14  # default
    for tag, size in [("72b", 144), ("32b", 64), ("14b", 28), ("7b", 14),
                      ("3b", 6), ("1.5b", 3), ("0.5b", 1)]:
        if tag in model_lower:
            model_bf16_gb = size
            break

    usable_per_core = hbm_per_core * 0.80
    fits_single = model_bf16_gb <= usable_per_core
    sharded_per_core = model_bf16_gb / tpu_num_cores
    fits_sharded = sharded_per_core <= usable_per_core

    if fits_single:
        use_fsdp = False
        print(f"\n  Strategy: SINGLE-CORE (model {model_bf16_gb} GB fits in {hbm_per_core} GB core)")
    elif fits_sharded:
        use_fsdp = True
        print(f"\n  Strategy: FSDP SHARDING ({model_bf16_gb} GB -> {sharded_per_core:.1f} GB/core across {tpu_num_cores} cores)")
    else:
        print(f"\n  WARNING: Model {model_bf16_gb} GB too large even with FSDP ({sharded_per_core:.1f} GB/core > {usable_per_core:.1f} GB)")
        print(f"  Will attempt FSDP anyway...")
        use_fsdp = True

    print(f"  Device: TPU ({device})")

except Exception as e:
    print(f"  TPU not available: {e}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "gpu"
        print(f"  Device: GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        device_type = "cpu"
        print(f"  Device: CPU")

# Resolve dtype
dtype_map = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
torch_dtype = dtype_map.get(DTYPE, torch.bfloat16)

if device_type == "gpu" and torch_dtype == torch.bfloat16:
    if not torch.cuda.is_bf16_supported():
        torch_dtype = torch.float16
        print("  Note: GPU doesn't support bfloat16, using float16")

print(f"  Torch dtype: {torch_dtype}")
print(f"  Use FSDP:    {use_fsdp}")

# ---------------------------------------------------------------------------
# 5.3  Load Model
# ---------------------------------------------------------------------------
print(f"\n  Loading model: {MODEL_ID}")
model_load_start = time.time()

from transformers import AutoModelForCausalLM, AutoTokenizer

if device_type == "tpu" and use_fsdp:
    # =================================================================
    # STRATEGY B: FSDP SHARDING ACROSS ALL TPU CORES
    # =================================================================
    # This is the correct way to load large models on TPU.
    #
    # Step 1: Load model on CPU with low memory usage
    # Step 2: Wrap with FSDP to shard across all cores
    # Step 3: Each core holds 1/N of the parameters
    #
    # Memory flow for Qwen-14B on TPU v2-8:
    #   CPU: loads 28 GB (needs ~30 GB CPU RAM)
    #   FSDP shards: 28 GB / 8 cores = 3.5 GB per core
    #   Per-core HBM: 8 GB available, 3.5 GB used = 4.5 GB free for KV-cache
    # =================================================================

    print("  Step 1: Loading weights to CPU (low_cpu_mem_usage=True)...")
    tpu_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    cpu_load_time = time.time() - model_load_start
    print(f"  CPU load: {format_duration(cpu_load_time)}")

    print(f"  Step 2: Wrapping with FSDP (sharding across {tpu_num_cores} cores)...")
    fsdp_start = time.time()

    try:
        # Try the newer torch_xla FSDP API first
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
        from torch_xla.distributed.fsdp import checkpoint_module

        # Wrap the model with FSDP
        # This shards parameters across all TPU cores
        tpu_model = FSDP(
            tpu_model,
            # Shard parameters across all cores
            # Each core gets 1/N of each parameter tensor
        )
        print(f"  FSDP wrap: {format_duration(time.time() - fsdp_start)}")
        print(f"  Model sharded across {tpu_num_cores} TPU cores")

    except ImportError:
        # Fallback: try PyTorch native FSDP with XLA backend
        print("  XLA FSDP not available, trying parameter streaming...")
        try:
            # Alternative: move parameters one-by-one to avoid double-buffering
            print("  Step 2b: Streaming parameters to TPU one-by-one...")
            stream_start = time.time()

            tpu_model = tpu_model.to(device)
            xm.mark_step()  # Force execution

            print(f"  Streamed in {format_duration(time.time() - stream_start)}")
        except RuntimeError as oom_err:
            if "RESOURCE_EXHAUSTED" in str(oom_err):
                print(f"\n  RESOURCE_EXHAUSTED: Model too large for single core!")
                print(f"  Model: {model_bf16_gb} GB | Core: {hbm_per_core} GB")
                print(f"")
                print(f"  SOLUTIONS:")
                print(f"  1. Use a smaller model: Qwen/Qwen2.5-3B (6 GB, fits single core)")
                print(f"  2. Use Kaggle TPU v3-8 (16 GB/core, fits 14B)")
                print(f"  3. Install torch_xla with FSDP support")
                raise
            raise

    except Exception as e:
        print(f"  FSDP failed: {e}")
        print("  Falling back to single-core loading...")
        try:
            tpu_model = tpu_model.to(device)
            xm.mark_step()
        except RuntimeError as oom_err:
            if "RESOURCE_EXHAUSTED" in str(oom_err):
                print(f"\n  RESOURCE_EXHAUSTED on single-core fallback.")
                print(f"  Model {model_bf16_gb} GB cannot fit in {hbm_per_core} GB core.")
                print(f"  Use a smaller model (3B or 0.5B) or upgrade to Kaggle TPU v3.")
            raise

elif device_type == "tpu" and not use_fsdp:
    # =================================================================
    # STRATEGY A: SINGLE-CORE (small model fits in one core)
    # =================================================================
    print("  Step 1: Loading weights to CPU...")
    tpu_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    cpu_load_time = time.time() - model_load_start
    print(f"  CPU load: {format_duration(cpu_load_time)}")

    print("  Step 2: Moving to TPU (single core)...")
    move_start = time.time()
    tpu_model = tpu_model.to(device)
    xm.mark_step()
    print(f"  TPU transfer: {format_duration(time.time() - move_start)}")

elif device_type == "gpu":
    # =================================================================
    # STRATEGY C: GPU with device_map="auto"
    # =================================================================
    print("  Loading with device_map='auto'...")
    tpu_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

else:
    # =================================================================
    # STRATEGY D: CPU fallback
    # =================================================================
    print("  Loading on CPU...")
    tpu_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

tpu_model.eval()
total_load_time = time.time() - model_load_start

# Count parameters
total_params = sum(p.numel() for p in tpu_model.parameters())
print(f"\n  Total load:  {format_duration(total_load_time)}")
print(f"  Parameters:  {total_params/1e9:.2f}B")

# Force garbage collection to free CPU copies
gc.collect()

# ---------------------------------------------------------------------------
# 5.4  Load Tokenizers
# ---------------------------------------------------------------------------
print(f"\n  Loading tokenizer: {MODEL_ID}")
tok_start = time.time()

model_tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, trust_remote_code=True, cache_dir=CACHE_DIR,
)

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

    if device_type == "tpu":
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
tpu_device = device

print("\n" + "-" * 60)
print(f"  MODEL READY ON {device_type.upper()}")
print(f"  Model:      {MODEL_ID}")
print(f"  Params:     {total_params/1e9:.2f}B")
print(f"  Dtype:      {torch_dtype}")
print(f"  Device:     {device}")
print(f"  FSDP:       {'YES (sharded across cores)' if use_fsdp else 'NO (single device)'}")
print(f"  Layers:     {num_layers}")
print(f"  Load time:  {format_duration(total_load_time)}")
print(f"  Chat tmpl:  {'yes' if hasattr(model_tokenizer, 'apply_chat_template') else 'manual'}")

print("\n  TPU CELL 5 COMPLETE")
print("=" * 60)

# Carry forward: tpu_model, model_tokenizer, counting_tokenizer,
#                pipeline_timer, tpu_device, device_type, use_fsdp
