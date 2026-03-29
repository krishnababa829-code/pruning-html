#!/usr/bin/env python3
"""
=============================================================================
  AirLLM Kaggle Pipeline  —  Run 70B+ Models on Free GPU
=============================================================================

Runs Qwen2.5-72B-Instruct (or any large model) on a Kaggle T4/P100 GPU
using AirLLM's layer-by-layer inference. Only ONE transformer layer is
loaded into VRAM at a time, so even 70B-parameter models fit in 16 GB.

Pipeline:
  1. Download 100 rows from williambrach/html-description-content
  2. Prune invisible HTML (script, style, svg, comments, data-*)
  3. Save newdataset-1.csv  (page_id, html)
  4. Run 3-task prompt through the large model via AirLLM
  5. Enforce 8000 Qwen-token limit per row
  6. Save newdataset-2.csv  (page_id, html, response)

Kaggle Setup (paste into first cell):
  !pip install -q airllm datasets transformers pandas beautifulsoup4 lxml accelerate
  !python airllm_kaggle_pipeline.py

Local Setup:
  pip install -r requirements_airllm.txt
  python airllm_kaggle_pipeline.py

=============================================================================
"""

import os
import re
import gc
import csv
import sys
import json
import time
import logging
import hashlib
import platform
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("airllm-pipeline")


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                         ║
# ╚═════════════════════════════════════════════════════════════════════════╝
@dataclass
class PipelineConfig:
    """Central configuration — edit these values to customize the pipeline."""

    # --- Model Selection ---------------------------------------------------
    # Tier 1 (Best):  Qwen/Qwen2.5-72B-Instruct
    # Tier 2:         meta-llama/Llama-3.1-70B-Instruct
    # Tier 3 (MoE):   mistralai/Mixtral-8x22B-Instruct-v0.1
    model_id: str = "Qwen/Qwen2.5-72B-Instruct"

    # --- Tokenizer (for token counting & truncation) -----------------------
    tokenizer_id: str = "Qwen/Qwen2.5-7B-Instruct"

    # --- Dataset -----------------------------------------------------------
    source_dataset: str = "williambrach/html-description-content"
    row_count: int = 100

    # --- Token Budget ------------------------------------------------------
    max_tokens_per_row: int = 8000
    max_new_tokens: int = 4096       # generation length
    html_budget_ratio: float = 0.40  # 40% HTML, 60% response

    # --- Generation --------------------------------------------------------
    temperature: float = 0.3
    top_p: float = 0.9
    repetition_penalty: float = 1.05

    # --- Output Files ------------------------------------------------------
    output_file_1: str = "newdataset-1.csv"
    output_file_2: str = "newdataset-2.csv"
    checkpoint_file: str = ".airllm_checkpoint.json"

    # --- AirLLM Settings ---------------------------------------------------
    compression: str = "4bit"        # '4bit', '8bit', or None
    cache_dir: str = "/kaggle/temp/model_cache"  # Kaggle scratch space
    prefetch_layers: int = 1         # prefetch next N layers

    # --- Retry / Rate Limiting ---------------------------------------------
    retry_limit: int = 3
    retry_delay: float = 5.0


CFG = PipelineConfig()


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  SYSTEM PROMPT  (3-Task Pipeline)                                      ║
# ╚═════════════════════════════════════════════════════════════════════════╝
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


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  ENVIRONMENT DETECTION                                                 ║
# ╚═════════════════════════════════════════════════════════════════════════╝
def detect_environment() -> Dict[str, Any]:
    """Auto-detect Kaggle, GPU availability, and memory."""
    env = {
        "is_kaggle": os.path.exists("/kaggle/working"),
        "platform": platform.system(),
        "python": platform.python_version(),
        "gpu_available": False,
        "gpu_name": "N/A",
        "gpu_vram_gb": 0.0,
    }

    try:
        import torch
        if torch.cuda.is_available():
            env["gpu_available"] = True
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_mem / (1024**3), 1
            )
    except ImportError:
        pass

    log.info("Environment: %s", json.dumps(env, indent=2))
    return env


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  HTML PRUNING ENGINE                                                   ║
# ╚═════════════════════════════════════════════════════════════════════════╝
class HTMLPruner:
    """Strips invisible/non-structural tags while preserving all visible HTML.

    Removes:  <style>, <script>, <noscript>, <svg>, <path>, <meta>, <link>,
              <iframe>, <object>, <embed>, HTML comments, class/style/on*/data-*
    Keeps:    <p>, <table>, <tr>, <td>, <th>, <h1>-<h6>, <b>, <i>, <a>, <ul>,
              <ol>, <li>, <code>, <pre>, <blockquote>, <img>, <br>, <hr>, etc.
    """

    INVISIBLE_TAGS = frozenset([
        "style", "script", "noscript", "meta", "link",
        "iframe", "object", "embed", "svg", "path",
        "head",  # entire <head> is non-visible
    ])

    STRIP_ATTRS = frozenset([
        "class", "style", "onclick", "onload", "onmouseover",
        "onfocus", "onblur", "onchange", "onsubmit", "onerror",
        "onkeydown", "onkeyup", "onkeypress",
    ])

    @classmethod
    def prune(cls, raw_html: str) -> str:
        """Clean HTML: remove invisible elements, keep structural content."""
        if not raw_html or not isinstance(raw_html, str):
            return ""

        from bs4 import BeautifulSoup, Comment

        soup = BeautifulSoup(raw_html, "lxml")

        # 1. Remove invisible tags entirely
        for tag_name in cls.INVISIBLE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # 2. Remove HTML comments
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            comment.extract()

        # 3. Strip noisy attributes (class, style, on*, data-*)
        for tag in soup.find_all(True):
            removable = [
                attr for attr in tag.attrs
                if attr in cls.STRIP_ATTRS or attr.startswith("data-")
            ]
            for attr in removable:
                del tag[attr]

        # 4. Remove empty tags that add no content (except self-closing ones)
        self_closing = frozenset(["br", "hr", "img", "input"])
        for tag in soup.find_all(True):
            if (
                tag.name not in self_closing
                and not tag.get_text(strip=True)
                and not tag.find_all(["img", "br", "hr", "input", "table"])
            ):
                tag.decompose()

        # 5. Collapse whitespace
        cleaned = str(soup)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        return cleaned.strip()


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  TOKEN MANAGER                                                         ║
# ╚═════════════════════════════════════════════════════════════════════════╝
class TokenManager:
    """Handles token counting and budget enforcement using the Qwen tokenizer."""

    def __init__(self, tokenizer_id: str = CFG.tokenizer_id):
        from transformers import AutoTokenizer
        log.info("Loading tokenizer: %s", tokenizer_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id, trust_remote_code=True
        )

    def count(self, text: str) -> int:
        """Count tokens in a string."""
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within max_tokens."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        return self.tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)

    def enforce_row_budget(
        self,
        page_id: str,
        html: str,
        response: str,
        max_tokens: int = CFG.max_tokens_per_row,
    ) -> Tuple[str, str]:
        """Ensure page_id + html + response fits within the token budget.
        Allocates 40% to HTML, 60% to response."""
        overhead = self.count(str(page_id)) + 20
        budget = max_tokens - overhead

        html_tokens = self.count(html)
        resp_tokens = self.count(response)

        if html_tokens + resp_tokens <= budget:
            return html, response

        html_budget = int(budget * CFG.html_budget_ratio)
        resp_budget = budget - html_budget

        html = self.truncate(html, html_budget)
        response = self.truncate(response, resp_budget)
        return html, response


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  CHECKPOINT MANAGER  (Resume interrupted runs)                         ║
# ╚═════════════════════════════════════════════════════════════════════════╝
class CheckpointManager:
    """Save/load progress so interrupted Kaggle sessions can resume."""

    def __init__(self, path: str = CFG.checkpoint_file):
        self.path = Path(path)

    def save(self, completed_indices: List[int], responses: Dict[int, str]):
        """Persist progress to disk."""
        data = {
            "completed": completed_indices,
            "responses": {str(k): v for k, v in responses.items()},
            "timestamp": time.time(),
        }
        self.path.write_text(json.dumps(data, ensure_ascii=False))
        log.info("Checkpoint saved: %d rows completed", len(completed_indices))

    def load(self) -> Tuple[List[int], Dict[int, str]]:
        """Load previous progress if available."""
        if not self.path.exists():
            return [], {}
        try:
            data = json.loads(self.path.read_text())
            completed = data.get("completed", [])
            responses = {int(k): v for k, v in data.get("responses", {}).items()}
            log.info("Checkpoint loaded: %d rows already completed", len(completed))
            return completed, responses
        except Exception as e:
            log.warning("Checkpoint corrupted, starting fresh: %s", e)
            return [], {}

    def clear(self):
        """Remove checkpoint file after successful completion."""
        if self.path.exists():
            self.path.unlink()


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  AirLLM INFERENCE ENGINE                                               ║
# ╚═════════════════════════════════════════════════════════════════════════╝
class AirLLMEngine:
    """Layer-by-layer inference using AirLLM.

    How AirLLM works:
      - Splits a large model (e.g. 72B params) into individual layers
      - Loads only ONE layer into GPU VRAM at a time
      - Streams input through each layer sequentially
      - Peak VRAM usage ~ size of 1 layer (~500MB-2GB) instead of full model
      - Supports 4-bit/8-bit compression to further reduce memory

    This means a 72B model that normally needs ~140GB VRAM can run on
    a Kaggle T4 (16GB) or even a P100 (16GB).
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Initialize AirLLM with the configured model."""
        log.info("Initializing AirLLM with model: %s", self.config.model_id)
        log.info("Compression: %s | Cache: %s", self.config.compression, self.config.cache_dir)

        # Ensure cache directory exists
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

        try:
            from airllm import AutoModel as AirAutoModel

            model_kwargs = {
                "cache_dir": self.config.cache_dir,
                "trust_remote_code": True,
            }

            # Apply compression
            if self.config.compression == "4bit":
                model_kwargs["compression"] = self.config.compression
            elif self.config.compression == "8bit":
                model_kwargs["compression"] = self.config.compression

            self.model = AirAutoModel.from_pretrained(
                self.config.model_id, **model_kwargs
            )
            log.info("AirLLM model loaded successfully")

        except ImportError:
            log.error(
                "AirLLM not installed. Run:\n"
                "  pip install airllm\n"
                "Or in Kaggle: !pip install -q airllm"
            )
            raise
        except Exception as e:
            log.error("Failed to load model: %s", e)
            raise

        # Load tokenizer separately for prompt formatting
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
            cache_dir=self.config.cache_dir,
        )

    def _build_prompt(self, html_content: str) -> str:
        """Build the full chat prompt using the model's chat template."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": html_content},
        ]

        # Use the tokenizer's chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback: manual Qwen/ChatML format
            prompt = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{html_content}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        return prompt

    def generate(self, html_content: str) -> Optional[str]:
        """Generate a response for the given HTML content."""
        import torch

        prompt = self._build_prompt(html_content)

        # Tokenize
        input_ids = self.tokenizer(
            prompt, return_tensors="pt"
        ).input_ids

        for attempt in range(1, self.config.retry_limit + 1):
            try:
                log.info(
                    "  Generating (attempt %d/%d, input_tokens=%d)...",
                    attempt, self.config.retry_limit, input_ids.shape[1],
                )

                # AirLLM generation
                generation_output = self.model.generate(
                    input_ids,
                    max_new_tokens=self.config.max_new_tokens,
                    use_cache=True,
                    return_dict_in_generate=True,
                )

                # Decode only the NEW tokens (skip the prompt)
                output_ids = generation_output.sequences[0]
                new_token_ids = output_ids[input_ids.shape[1]:]
                response = self.tokenizer.decode(
                    new_token_ids, skip_special_tokens=True
                )

                # Force garbage collection after each generation
                del generation_output, output_ids, new_token_ids
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                return response.strip()

            except torch.cuda.OutOfMemoryError:
                log.warning("  OOM on attempt %d — clearing cache and retrying", attempt)
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(self.config.retry_delay)

            except Exception as e:
                log.warning("  Attempt %d failed: %s", attempt, e)
                if attempt < self.config.retry_limit:
                    time.sleep(self.config.retry_delay * attempt)

        return None


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  DATASET MANAGER                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════╝
class DatasetManager:
    """Handles downloading, column detection, and saving."""

    @staticmethod
    def download(source: str = CFG.source_dataset, n_rows: int = CFG.row_count) -> pd.DataFrame:
        """Download the first n_rows from the HuggingFace dataset."""
        from datasets import load_dataset

        log.info("Downloading: %s (first %d rows)", source, n_rows)
        ds = load_dataset(source, split=f"train[:{n_rows}]")
        df = ds.to_pandas()

        # Auto-detect columns
        col_map = {}
        for col in df.columns:
            lower = col.lower().strip()
            if "page" in lower and "id" in lower:
                col_map["page_id"] = col
            if lower == "html":
                col_map["html"] = col

        if "page_id" not in col_map or "html" not in col_map:
            log.warning("Auto-detect failed. Columns: %s. Using first two.", list(df.columns))
            col_map["page_id"] = df.columns[0]
            col_map["html"] = df.columns[1]

        df = df[[col_map["page_id"], col_map["html"]]].copy()
        df.columns = ["page_id", "html"]
        log.info("Downloaded %d rows", len(df))
        return df

    @staticmethod
    def save(df: pd.DataFrame, path: str):
        """Save DataFrame to CSV with proper quoting."""
        df.to_csv(path, index=False, quoting=csv.QUOTE_ALL)
        log.info("Saved: %s (%d rows, columns: %s)", path, len(df), list(df.columns))


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  RESPONSE VALIDATOR                                                    ║
# ╚═════════════════════════════════════════════════════════════════════════╝
class ResponseValidator:
    """Validates that model output contains all 3 required task sections."""

    REQUIRED_MARKERS = ["===TASK 1===", "===TASK 2===", "===TASK 3==="]

    @classmethod
    def validate(cls, response: str) -> Dict[str, Any]:
        """Check response structure and return validation report."""
        if not response:
            return {"valid": False, "reason": "empty response", "tasks_found": 0}

        tasks_found = sum(1 for m in cls.REQUIRED_MARKERS if m in response)
        has_user_query = "USER_QUERY:" in response

        if tasks_found == 3 and has_user_query:
            return {"valid": True, "reason": "all tasks present", "tasks_found": 3}

        return {
            "valid": False,
            "reason": f"missing tasks ({tasks_found}/3), USER_QUERY={'yes' if has_user_query else 'no'}",
            "tasks_found": tasks_found,
        }

    @classmethod
    def extract_tasks(cls, response: str) -> Dict[str, str]:
        """Extract individual task outputs from the response."""
        result = {"task1": "", "task2": "", "task3": ""}
        if not response:
            return result

        parts = re.split(r"===TASK \d+===", response)
        if len(parts) >= 4:
            result["task1"] = parts[1].strip()  # Markdown
            result["task2"] = parts[2].strip()  # Signpost
            result["task3"] = parts[3].strip()  # Router queries
        elif len(parts) >= 3:
            result["task1"] = parts[1].strip()
            result["task2"] = parts[2].strip()
        elif len(parts) >= 2:
            result["task1"] = parts[1].strip()

        return result


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  MAIN PIPELINE                                                         ║
# ╚═════════════════════════════════════════════════════════════════════════╝
def run_pipeline():
    """Execute the full AirLLM pipeline."""

    start_time = time.time()
    log.info("="*60)
    log.info("  AirLLM KAGGLE PIPELINE")
    log.info("  Model: %s", CFG.model_id)
    log.info("  Compression: %s", CFG.compression)
    log.info("  Rows: %d | Max tokens/row: %d", CFG.row_count, CFG.max_tokens_per_row)
    log.info("="*60)

    # --- Environment Detection ---
    env = detect_environment()

    if not env["gpu_available"]:
        log.warning(
            "No GPU detected! AirLLM works best with a GPU.\n"
            "On Kaggle: Settings > Accelerator > GPU T4 x2\n"
            "Continuing on CPU (will be very slow)..."
        )

    # --- Step 1: Download Dataset ---
    log.info("\n--- STEP 1: Download Dataset ---")
    df = DatasetManager.download()

    # --- Step 2: Prune HTML ---
    log.info("\n--- STEP 2: Prune HTML ---")
    pruner = HTMLPruner()
    original_sizes = df["html"].str.len().tolist()
    df["html"] = df["html"].apply(HTMLPruner.prune)
    pruned_sizes = df["html"].str.len().tolist()

    total_original = sum(original_sizes)
    total_pruned = sum(pruned_sizes)
    reduction = (1 - total_pruned / max(total_original, 1)) * 100
    log.info(
        "HTML pruning: %s chars -> %s chars (%.1f%% reduction)",
        f"{total_original:,}", f"{total_pruned:,}", reduction,
    )

    # --- Step 3: Save newdataset-1.csv ---
    log.info("\n--- STEP 3: Save newdataset-1.csv ---")
    DatasetManager.save(df, CFG.output_file_1)

    # --- Step 4: Initialize Components ---
    log.info("\n--- STEP 4: Initialize AirLLM + Tokenizer ---")
    token_mgr = TokenManager()
    checkpoint = CheckpointManager()
    validator = ResponseValidator()

    # Pre-truncate oversized HTML
    html_budget = int(CFG.max_tokens_per_row * CFG.html_budget_ratio)
    for idx in df.index:
        if token_mgr.count(df.at[idx, "html"]) > html_budget:
            df.at[idx, "html"] = token_mgr.truncate(df.at[idx, "html"], html_budget)

    # Load AirLLM engine
    engine = AirLLMEngine(CFG)

    # --- Step 5: Generate Responses (with checkpoint/resume) ---
    log.info("\n--- STEP 5: Generate Responses ---")
    completed_indices, responses = checkpoint.load()

    stats = {"success": 0, "failed": 0, "skipped": len(completed_indices), "valid": 0}

    for idx, row in df.iterrows():
        if idx in completed_indices:
            log.info("Row %d/%d — skipped (checkpoint)", idx + 1, len(df))
            continue

        log.info(
            "Row %d/%d | page_id=%s | html_chars=%d",
            idx + 1, len(df), row["page_id"], len(row["html"]),
        )

        response = engine.generate(row["html"])

        if response is None:
            response = "[ERROR] Generation failed after retries."
            stats["failed"] += 1
        else:
            stats["success"] += 1
            # Validate
            validation = validator.validate(response)
            if validation["valid"]:
                stats["valid"] += 1
            else:
                log.warning("  Validation: %s", validation["reason"])

        responses[idx] = response
        completed_indices.append(idx)

        # Save checkpoint every 5 rows
        if len(completed_indices) % 5 == 0:
            checkpoint.save(completed_indices, responses)

    # --- Step 6: Assemble & Enforce Token Budget ---
    log.info("\n--- STEP 6: Enforce Token Budget ---")
    df["response"] = df.index.map(lambda i: responses.get(i, "[ERROR] No response"))

    for idx in df.index:
        html_trimmed, resp_trimmed = token_mgr.enforce_row_budget(
            page_id=df.at[idx, "page_id"],
            html=df.at[idx, "html"],
            response=df.at[idx, "response"],
        )
        df.at[idx, "html"] = html_trimmed
        df.at[idx, "response"] = resp_trimmed

    # --- Step 7: Save newdataset-2.csv ---
    log.info("\n--- STEP 7: Save newdataset-2.csv ---")
    DatasetManager.save(df, CFG.output_file_2)
    checkpoint.clear()

    # --- Summary ---
    elapsed = time.time() - start_time
    log.info("\n" + "="*60)
    log.info("  PIPELINE COMPLETE")
    log.info("="*60)
    log.info("  Model:        %s", CFG.model_id)
    log.info("  Compression:  %s", CFG.compression)
    log.info("  Time:         %.1f minutes", elapsed / 60)
    log.info("  Rows:         %d total", len(df))
    log.info("  Success:      %d", stats["success"])
    log.info("  Valid (3/3):  %d", stats["valid"])
    log.info("  Failed:       %d", stats["failed"])
    log.info("  Skipped:      %d (from checkpoint)", stats["skipped"])
    log.info("  Output 1:     %s", CFG.output_file_1)
    log.info("  Output 2:     %s", CFG.output_file_2)

    # Token distribution stats
    token_counts = []
    for _, row in df.iterrows():
        total = (
            token_mgr.count(str(row["page_id"]))
            + token_mgr.count(row["html"])
            + token_mgr.count(row["response"])
        )
        token_counts.append(total)

    if token_counts:
        log.info(
            "  Tokens:       min=%d  max=%d  mean=%d",
            min(token_counts), max(token_counts),
            sum(token_counts) // len(token_counts),
        )
    log.info("="*60)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  ENTRY POINT                                                           ║
# ╚═════════════════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    run_pipeline()
