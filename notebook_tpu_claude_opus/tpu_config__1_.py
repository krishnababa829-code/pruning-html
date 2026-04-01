#!/usr/bin/env python3
"""
=============================================================================
  TPU CONFIG  -  Shared Configuration for TPU Notebook Cells
=============================================================================
  Central configuration. Every other cell imports from here.

  IMPORTANT: This pipeline does NOT use AirLLM.
  AirLLM is CUDA-only (GPU). TPU uses XLA, a completely different backend.

  TPU MEMORY ARCHITECTURE (critical to understand):
  -----------------------------------------------
  TPU v2-8:  8 cores x  8 GB HBM each = 64 GB total
  TPU v3-8:  8 cores x 16 GB HBM each = 128 GB total

  BUT: xm.xla_device() -> xla:0 = ONE core only!
  model.to(xla:0) loads the ENTIRE model onto ONE core.

  To use all cores, you MUST use FSDP (Fully Sharded Data Parallel).
  Cell 5 handles this automatically based on model size.

  Model sizing guide (per-core, NOT total):
  -----------------------------------------
  TPU v2-8 (8 GB/core):
    Single core: Qwen-3B (6 GB) or smaller
    FSDP 8-way:  Qwen-32B (64 GB / 8 = 8 GB/core, tight)
                 Qwen-14B (28 GB / 8 = 3.5 GB/core, comfortable)
                 Qwen-7B  (14 GB / 8 = 1.75 GB/core, easy)

  TPU v3-8 (16 GB/core):
    Single core: Qwen-7B (14 GB) or smaller
    FSDP 8-way:  Qwen-72B (144 GB / 8 = 18 GB/core, tight)
                 Qwen-32B (64 GB / 8 = 8 GB/core, comfortable)

  Edit this cell FIRST before running anything else.
=============================================================================
"""

import os

# =============================================================================
#  USER SETTINGS
# =============================================================================

ROW_COUNT = 100  # <-- CHANGE THIS: number of rows to process

# Model selection (see sizing guide above)
# For Colab TPU v2-8 with FSDP: up to 32B (tight) or 14B (comfortable)
# For Kaggle TPU v3-8 with FSDP: up to 72B (tight) or 32B (comfortable)
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"  # <-- CHANGE THIS

# Precision: bfloat16 is TPU-native and gives best performance
DTYPE = "bfloat16"  # <-- CHANGE THIS

# Token budget per row in final CSV
MAX_TOKENS_PER_ROW = 8000
MAX_NEW_TOKENS = 4096

# Generation parameters
TEMPERATURE = 0.3
TOP_P = 0.9

# =============================================================================
#  DATASET SETTINGS
# =============================================================================
SOURCE_DATASET = "williambrach/html-description-content"
TOKENIZER_ID = "Qwen/Qwen2.5-7B-Instruct"
HTML_BUDGET_RATIO = 0.40

# =============================================================================
#  ENVIRONMENT DETECTION
# =============================================================================
IS_COLAB = os.path.exists("/content") and not os.path.exists("/kaggle")
IS_KAGGLE = os.path.exists("/kaggle/working")

# =============================================================================
#  PATHS
# =============================================================================
REPO_SLUG = "phanirishindra/pruning-html"
REPO_NAME = REPO_SLUG.split("/")[-1]
REPO_URL = f"https://github.com/{REPO_SLUG}.git"

if IS_COLAB:
    WORK_DIR = f"/content/{REPO_NAME}"
    CACHE_DIR = "/content/model_cache"
elif IS_KAGGLE:
    WORK_DIR = f"/kaggle/working/{REPO_NAME}"
    CACHE_DIR = "/kaggle/temp/model_cache"
else:
    WORK_DIR = "."
    CACHE_DIR = "./model_cache"

_OUTPUT_FILE_1 = "newdataset-1-tpu.csv"
_OUTPUT_FILE_2 = "newdataset-2-tpu.csv"
_CHECKPOINT_FILE = ".tpu_checkpoint.json"
_TIMING_REPORT_FILE = "timing_report_tpu.json"

OUTPUT_FILE_1 = os.path.join(WORK_DIR, _OUTPUT_FILE_1)
OUTPUT_FILE_2 = os.path.join(WORK_DIR, _OUTPUT_FILE_2)
CHECKPOINT_FILE = os.path.join(WORK_DIR, _CHECKPOINT_FILE)
TIMING_REPORT_FILE = os.path.join(WORK_DIR, _TIMING_REPORT_FILE)

# =============================================================================
#  SYSTEM PROMPT  (identical to GPU version)
# =============================================================================
SYSTEM_PROMPT = """CRITICAL DIRECTIVES:
- Do NOT output any conversational text or greetings.
- VERBATIM TRANSLATION: You must not summarize, paraphrase, hallucinate or omit a single word from the user's HTML.
- Format your output EXACTLY with the dividers shown below.

For every raw HTML chunk the user provides, execute these 3 Tasks:

TASK 1: THE CLEANER (HTML -> Markdown)
Translate the HTML into strict Markdown. Preserve all |---| tables, code blocks, hyperlinks and links. Do NOT summarize.

TASK 2: THE INDEXER (Markdown -> Signpost)
Read the Markdown you just created. Write a Dense Signpost for it.
Format exactly: [Core Theme] + [Key Entities] + [Questions Answered] (Max 70 words).

TASK 3: THE ROUTER (Query Deconstruction)
Create EXACTLY THREE synthetic examples of a user asking a highly scrambled, slang-filled, multi-part query that matches the chunk you just processed.
For EACH of the three examples, you must output a separate USER_QUERY and ASSISTANT block.
The Assistant output must start with a <think> tag explaining the slang-to-entity translation, followed by a JSON array ["chunk_xyz"].

OUTPUT FORMAT:
===TASK 1===
<PERFECT MARKDOWN HERE>
===TASK 2===
[Theme] + [Entities] + [Answers]
===TASK 3===
USER_QUERY: <First Scrambled Slang Query>
ASSISTANT:
<think>...</think>
["chunk_xyz"]

USER_QUERY: <Second Scrambled Slang Query>
ASSISTANT:
<think>...</think>
["chunk_xyz"]

USER_QUERY: <Third Scrambled Slang Query>
ASSISTANT:
<think>...</think>
["chunk_xyz"]"""

# =============================================================================
#  PRINT SUMMARY
# =============================================================================
if __name__ == "__main__" or True:
    print("\n" + "=" * 60)
    print("  TPU NOTEBOOK CONFIG LOADED")
    print("=" * 60)
    print(f"  Environment:    {'Colab' if IS_COLAB else 'Kaggle' if IS_KAGGLE else 'Local'}")
    print(f"  Model:          {MODEL_ID}")
    print(f"  Dtype:          {DTYPE} (TPU-native: bfloat16)")
    print(f"  Rows:           {ROW_COUNT}")
    print(f"  Max tokens/row: {MAX_TOKENS_PER_ROW}")
    print(f"  Backend:        PyTorch/XLA (NOT AirLLM)")
    print(f"  Work dir:       {WORK_DIR}")
    print(f"  Cache dir:      {CACHE_DIR}")
    print(f"")
    print(f"  TPU MEMORY WARNING:")
    print(f"    TPU v2-8 = 8 cores x 8 GB = 64 GB total")
    print(f"    BUT xm.xla_device() = ONE core = 8 GB only!")
    print(f"    Models > 6 GB need FSDP sharding (Cell 5 handles this).")
    print("=" * 60)
