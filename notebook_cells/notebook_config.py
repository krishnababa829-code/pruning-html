#!/usr/bin/env python3
"""
=============================================================================
  NOTEBOOK CONFIG  -  Shared Configuration for All Cells
=============================================================================
  Edit this cell FIRST before running anything else.
  All other cells import from here.
=============================================================================
"""

import os

# =============================================================================
#  USER SETTINGS  (Change these to your needs)
# =============================================================================

# How many rows to download from the HuggingFace dataset
# Change this to any number you want: 10, 50, 100, 500, etc.
ROW_COUNT = 100  # <-- CHANGE THIS

# Model to use (ranked by quality for this task)
# Option 1 (BEST):  "Qwen/Qwen2.5-72B-Instruct"     - 80 layers, best HTML fidelity
# Option 2:         "meta-llama/Llama-3.1-70B-Instruct" - 80 layers, strong structured output
# Option 3:         "Qwen/Qwen2.5-32B-Instruct"      - 64 layers, faster, still good
# Option 4:         "Qwen/Qwen2.5-7B-Instruct"       - 32 layers, fastest, lower quality
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"  # <-- CHANGE THIS

# Compression (reduces VRAM usage)
# "4bit" = smallest memory footprint (recommended for free T4)
# "8bit" = better quality, more memory
# None   = no compression (needs lots of VRAM)
COMPRESSION = "4bit"  # <-- CHANGE THIS

# Token budget per row in final CSV
MAX_TOKENS_PER_ROW = 8000

# Max new tokens the model can generate per row
MAX_NEW_TOKENS = 4096

# Generation parameters
TEMPERATURE = 0.3
TOP_P = 0.9

# =============================================================================
#  DATASET SETTINGS
# =============================================================================
SOURCE_DATASET = "williambrach/html-description-content"
TOKENIZER_ID = "Qwen/Qwen2.5-7B-Instruct"  # For token counting
HTML_BUDGET_RATIO = 0.40  # 40% HTML, 60% response in final CSV

# =============================================================================
#  OUTPUT FILES  (absolute paths resolved below after WORK_DIR is set)
# =============================================================================
_OUTPUT_FILE_1 = "newdataset-1.csv"   # page_id, html (pruned)
_OUTPUT_FILE_2 = "newdataset-2.csv"   # page_id, html, response
_CHECKPOINT_FILE = ".airllm_checkpoint.json"
_TIMING_REPORT_FILE = "timing_report.json"

# =============================================================================
#  PATHS  (auto-detected, no need to change)
# =============================================================================
IS_COLAB = os.path.exists("/content") and not os.path.exists("/kaggle")
IS_KAGGLE = os.path.exists("/kaggle/working")

if IS_COLAB:
    WORK_DIR = "/content/cruzesolutions-project"
    CACHE_DIR = "/content/model_cache"
elif IS_KAGGLE:
    WORK_DIR = "/kaggle/working/cruzesolutions-project"
    CACHE_DIR = "/kaggle/temp/model_cache"
else:
    WORK_DIR = "."
    CACHE_DIR = "./model_cache"

REPO_URL = "https://gitlab.com/cruzesolutions-group/cruzesolutions-project.git"

# Resolve output file paths to absolute using WORK_DIR
OUTPUT_FILE_1 = os.path.join(WORK_DIR, _OUTPUT_FILE_1)
OUTPUT_FILE_2 = os.path.join(WORK_DIR, _OUTPUT_FILE_2)
CHECKPOINT_FILE = os.path.join(WORK_DIR, _CHECKPOINT_FILE)
TIMING_REPORT_FILE = os.path.join(WORK_DIR, _TIMING_REPORT_FILE)

# =============================================================================
#  SYSTEM PROMPT  (3-Task Pipeline)
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
print("\n" + "=" * 60)
print("  NOTEBOOK CONFIG LOADED")
print("=" * 60)
print(f"  Environment:    {'Colab' if IS_COLAB else 'Kaggle' if IS_KAGGLE else 'Local'}")
print(f"  Model:          {MODEL_ID}")
print(f"  Compression:    {COMPRESSION}")
print(f"  Rows:           {ROW_COUNT}")
print(f"  Max tokens/row: {MAX_TOKENS_PER_ROW}")
print(f"  Work dir:       {WORK_DIR}")
print(f"  Cache dir:      {CACHE_DIR}")
print("=" * 60)
