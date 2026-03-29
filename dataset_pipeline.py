#!/usr/bin/env python3
"""
Dataset Processing Pipeline
============================
Downloads 100 rows from williambrach/html-description-content,
prunes invisible HTML tags, applies a 3-task prompt via
HuggingFace Inference API, enforces 8000 Qwen-token limit per row,
and saves newdataset-1.csv and newdataset-2.csv.

Usage:
    export HF_API_TOKEN="hf_..."
    python dataset_pipeline.py
"""

import os
import re
import csv
import time
import logging
from typing import Optional

import pandas as pd
from datasets import load_dataset
from bs4 import BeautifulSoup
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"          # HF Inference model
TOKENIZER_ID = "Qwen/Qwen2.5-7B-Instruct"        # For token counting
SOURCE_DATASET = "williambrach/html-description-content"
ROW_COUNT = 100
MAX_TOKENS_PER_ROW = 8000
OUTPUT_FILE_1 = "newdataset-1.csv"
OUTPUT_FILE_2 = "newdataset-2.csv"
RETRY_LIMIT = 3
RETRY_DELAY = 5  # seconds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
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
Output your results exactly like this. DO NOT print the task names or descriptions. ONLY use these exact text dividers:

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


# ---------------------------------------------------------------------------
# 1. HTML Pruning
# ---------------------------------------------------------------------------
def prune_html(raw_html: str) -> str:
    """Remove invisible tags (<style>, <script>, comments, meta, link, noscript)
    while keeping all structural HTML intact (<p>, <table>, <h1>, <b>, etc.)."""
    if not raw_html or not isinstance(raw_html, str):
        return ""

    soup = BeautifulSoup(raw_html, "lxml")

    # Tags that carry no visible content
    invisible_tags = [
        "style", "script", "noscript", "meta", "link",
        "iframe", "object", "embed", "svg", "path",
    ]
    for tag_name in invisible_tags:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove HTML comments
    from bs4 import Comment
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # Strip all class, style, onclick, data-* attributes (reduce noise)
    for tag in soup.find_all(True):
        attrs_to_remove = []
        for attr in tag.attrs:
            if attr in ("class", "style", "onclick", "onload", "onmouseover"):
                attrs_to_remove.append(attr)
            elif attr.startswith("data-"):
                attrs_to_remove.append(attr)
        for attr in attrs_to_remove:
            del tag[attr]

    # Collapse excessive whitespace
    cleaned = str(soup)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# 2. Token counting & truncation (Qwen tokenizer)
# ---------------------------------------------------------------------------
def load_tokenizer():
    """Load the Qwen tokenizer for accurate token counting."""
    log.info("Loading Qwen tokenizer: %s", TOKENIZER_ID)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)
    return tokenizer


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens using the Qwen tokenizer."""
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def truncate_to_token_limit(
    page_id: str,
    html_col: str,
    response_col: str,
    tokenizer,
    max_tokens: int = MAX_TOKENS_PER_ROW,
) -> tuple[str, str]:
    """Ensure the combined row (page_id + html + response) fits within
    max_tokens. Trims the html first, then the response if still over."""
    overhead = count_tokens(str(page_id), tokenizer) + 20  # separators / CSV overhead
    budget = max_tokens - overhead

    html_tokens = count_tokens(html_col, tokenizer)
    resp_tokens = count_tokens(response_col, tokenizer)

    if html_tokens + resp_tokens <= budget:
        return html_col, response_col

    # Allocate 40 % to HTML, 60 % to response
    html_budget = int(budget * 0.4)
    resp_budget = budget - html_budget

    if html_tokens > html_budget:
        ids = tokenizer.encode(html_col, add_special_tokens=False)[:html_budget]
        html_col = tokenizer.decode(ids, skip_special_tokens=True)

    if resp_tokens > resp_budget:
        ids = tokenizer.encode(response_col, add_special_tokens=False)[:resp_budget]
        response_col = tokenizer.decode(ids, skip_special_tokens=True)

    return html_col, response_col


# ---------------------------------------------------------------------------
# 3. Download dataset
# ---------------------------------------------------------------------------
def download_dataset() -> pd.DataFrame:
    """Download the first ROW_COUNT rows with page_id and html columns."""
    log.info("Downloading dataset: %s (first %d rows)", SOURCE_DATASET, ROW_COUNT)
    ds = load_dataset(SOURCE_DATASET, split=f"train[:{ROW_COUNT}]")
    df = ds.to_pandas()

    # Identify the correct column names (case-insensitive search)
    col_map = {}
    for col in df.columns:
        lower = col.lower().strip()
        if "page" in lower and "id" in lower:
            col_map["page_id"] = col
        if lower == "html":
            col_map["html"] = col

    if "page_id" not in col_map or "html" not in col_map:
        log.warning("Could not auto-detect columns. Available: %s", list(df.columns))
        log.warning("Falling back to first two columns.")
        col_map["page_id"] = df.columns[0]
        col_map["html"] = df.columns[1]

    df = df[[col_map["page_id"], col_map["html"]]].copy()
    df.columns = ["page_id", "html"]
    log.info("Downloaded %d rows. Columns: %s", len(df), list(df.columns))
    return df


# ---------------------------------------------------------------------------
# 4. Apply prompt via HuggingFace Inference API
# ---------------------------------------------------------------------------
def apply_prompt(html_content: str, client: InferenceClient) -> Optional[str]:
    """Send pruned HTML to the model and return the response."""
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = client.chat_completion(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": html_content},
                ],
                max_tokens=4096,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as exc:
            log.warning("Attempt %d/%d failed: %s", attempt, RETRY_LIMIT, exc)
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY * attempt)
    return None


# ---------------------------------------------------------------------------
# 5. Main pipeline
# ---------------------------------------------------------------------------
def main():
    if not HF_API_TOKEN:
        log.error(
            "HF_API_TOKEN not set. Export it before running:\n"
            "  export HF_API_TOKEN='hf_...'\n"
            "Get a free token at https://huggingface.co/settings/tokens"
        )
        return

    # --- Step A: Download & prune ----------------------------------------
    df = download_dataset()

    log.info("Pruning HTML (removing <style>, <script>, comments, etc.)...")
    df["html"] = df["html"].apply(prune_html)

    # Save newdataset-1.csv (page_id, html)
    df.to_csv(OUTPUT_FILE_1, index=False, quoting=csv.QUOTE_ALL)
    log.info("Saved %s  (%d rows, columns: %s)", OUTPUT_FILE_1, len(df), list(df.columns))

    # --- Step B: Load tokenizer ------------------------------------------
    tokenizer = load_tokenizer()

    # Pre-truncate HTML that is already too large on its own
    html_solo_budget = int(MAX_TOKENS_PER_ROW * 0.4)
    for idx in df.index:
        html_text = df.at[idx, "html"]
        if count_tokens(html_text, tokenizer) > html_solo_budget:
            ids = tokenizer.encode(html_text, add_special_tokens=False)[:html_solo_budget]
            df.at[idx, "html"] = tokenizer.decode(ids, skip_special_tokens=True)

    # --- Step C: Apply prompt to each row --------------------------------
    client = InferenceClient(token=HF_API_TOKEN)
    responses = []

    for idx, row in df.iterrows():
        log.info("Processing row %d/%d  (page_id=%s)", idx + 1, len(df), row["page_id"])
        resp = apply_prompt(row["html"], client)
        if resp is None:
            resp = "[ERROR] Failed after retries."
        responses.append(resp)
        # Polite rate-limiting
        time.sleep(1)

    df["response"] = responses

    # --- Step D: Enforce 8000-token limit per row ------------------------
    log.info("Enforcing %d-token limit per row...", MAX_TOKENS_PER_ROW)
    for idx in df.index:
        html_trimmed, resp_trimmed = truncate_to_token_limit(
            page_id=df.at[idx, "page_id"],
            html_col=df.at[idx, "html"],
            response_col=df.at[idx, "response"],
            tokenizer=tokenizer,
        )
        df.at[idx, "html"] = html_trimmed
        df.at[idx, "response"] = resp_trimmed

    # --- Step E: Save newdataset-2.csv -----------------------------------
    df.to_csv(OUTPUT_FILE_2, index=False, quoting=csv.QUOTE_ALL)
    log.info("Saved %s  (%d rows, columns: %s)", OUTPUT_FILE_2, len(df), list(df.columns))

    # --- Summary ---------------------------------------------------------
    log.info("\n=== PIPELINE COMPLETE ===")
    log.info("%s -> %d rows  [page_id, html]", OUTPUT_FILE_1, len(df))
    log.info("%s -> %d rows  [page_id, html, response]", OUTPUT_FILE_2, len(df))

    # Token stats
    token_counts = []
    for _, row in df.iterrows():
        total = (
            count_tokens(str(row["page_id"]), tokenizer)
            + count_tokens(row["html"], tokenizer)
            + count_tokens(row["response"], tokenizer)
        )
        token_counts.append(total)
    log.info(
        "Token stats  ->  min=%d  max=%d  mean=%d",
        min(token_counts),
        max(token_counts),
        sum(token_counts) // len(token_counts),
    )


if __name__ == "__main__":
    main()
