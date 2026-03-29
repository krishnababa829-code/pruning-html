# Dataset Processing Pipeline

Downloads 100 rows from [`williambrach/html-description-content`](https://huggingface.co/datasets/williambrach/html-description-content), prunes invisible HTML, applies a 3-task prompt via HuggingFace Inference API, enforces an 8 000 Qwen-token limit per row, and produces two CSV files.

## Output Files

| File | Columns | Description |
|------|---------|-------------|
| `newdataset-1.csv` | `page_id`, `html` | Pruned HTML only |
| `newdataset-2.csv` | `page_id`, `html`, `response` | + model response column |

## Setup

```bash
pip install -r requirements.txt
```

Get a free HuggingFace token at <https://huggingface.co/settings/tokens> (needs Inference API access).

```bash
export HF_API_TOKEN="hf_..."
```

## Run

```bash
python dataset_pipeline.py
```

The script will:

1. Download 100 rows (`page_id`, `html`) and save `newdataset-1.csv`
2. Prune invisible tags (`<style>`, `<script>`, comments, `data-*` attributes)
3. Apply the 3-task prompt to each row via HuggingFace Inference API
4. Enforce 8 000 Qwen tokens per row (40 % HTML / 60 % response split)
5. Save `newdataset-2.csv` with the added `response` column

## Configuration

Edit the constants at the top of `dataset_pipeline.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `Qwen/Qwen2.5-72B-Instruct` | HF Inference model |
| `TOKENIZER_ID` | `Qwen/Qwen2.5-7B-Instruct` | Tokenizer for counting |
| `ROW_COUNT` | `100` | Rows to download |
| `MAX_TOKENS_PER_ROW` | `8000` | Token budget per row |
