# DocVL-IN — Vision-Language Model for Indian Document Understanding

Fine-tunes an open-source vision-language model (default: **Qwen2-VL-2B-Instruct**) to turn
photos/scans of Indian documents — invoices, government forms, ID cards — into structured
JSON, with a full pipeline covering data → training → evaluation → quantized serving →
retrieval, mirroring the lifecycle of a production VLM system.

Documents are a mix of **English + Hindi (Devanagari)**, since that's the realistic case for
an India-focused deployment, not a benchmark curated for English-only OCR.

## Why this project exists

Built to demonstrate hands-on experience across the full VLM lifecycle:

| Stage | What's implemented | File |
|---|---|---|
| Multimodal data pipeline | Synthetic doc generation, public dataset ingestion, dedup, quality filtering | `data/` |
| Fine-tuning | QLoRA fine-tune of a VLM on structured extraction | `training/finetune_lora.py` |
| Evaluation harness | Field-level F1, exact match, CER, automated regression report | `eval/eval_harness.py` |
| Inference optimization | 4-bit quantization + latency/throughput benchmarking | `inference/quantize.py` |
| Production serving | FastAPI extraction endpoint with batching | `inference/serve_api.py` |
| Retrieval / visual search | Embedding index over documents, "find similar doc" search | `retrieval/visual_search.py` |

## Requirements

This repo is a **scaffold you run on a GPU machine** (Colab Pro / Kaggle / RunPod / Lambda).
It was written and organized in an environment with no GPU/network access, so treat it as a
strong starting point — run it, watch what breaks, and fix it. That debugging trail (documented
in `TRAINING_LOG.md`, which you fill in as you go) is itself part of what the project is meant
to show: real training runs don't work on the first try.

```bash
pip install -r requirements.txt
```

A CUDA GPU with >= 16GB VRAM is recommended for LoRA fine-tuning of the 2B model
(T4 15GB works with 4-bit quant + small batch size; use an A10/A100 if fine-tuning the 7B variant).

## Pipeline

```bash
# 1. Build the dataset (synthetic + public sources, deduped & filtered)
python data/generate_synthetic.py --n 500 --out data/synthetic
python data/prepare_dataset.py --synthetic data/synthetic --out data/processed

# 2. Fine-tune
python training/finetune_lora.py --config training/config.yaml

# 3. Evaluate (baseline vs fine-tuned — run this BEFORE fine-tuning too, to get a baseline)
python eval/eval_harness.py --model Qwen/Qwen2-VL-2B-Instruct --data data/processed/test.jsonl --out eval/results_baseline.json
python eval/eval_harness.py --model training/output/checkpoint-final --data data/processed/test.jsonl --out eval/results_finetuned.json

# 4. Quantize + benchmark inference
python inference/quantize.py --model training/output/checkpoint-final --out training/output/quantized

# 5. Serve
uvicorn inference.serve_api:app --host 0.0.0.0 --port 8000

# 6. Build a visual search index over a folder of documents
python retrieval/visual_search.py build --docs data/processed/images --index retrieval/index.pkl
python retrieval/visual_search.py query --index retrieval/index.pkl --image samples/query.jpg
```

## Reporting results (fill this in after you run it)

Put a table like this in your README/GitHub/resume link — this comparison is the actual
portfolio artifact, not the raw accuracy number:

| Model | Field-level F1 | Exact Match | Latency (p50, ms) | Model size |
|---|---|---|---|---|
| Qwen2-VL-2B (zero-shot baseline) | — | — | — | 4.1 GB |
| Qwen2-VL-2B (LoRA fine-tuned) | — | — | — | 4.1 GB |
| + 4-bit quantized | — | — | — | ~1.2 GB |

## Project structure

```
docvl-in/
├── data/
│   ├── schema.py              # extraction JSON schemas (invoice / form / ID card)
│   ├── generate_synthetic.py  # synthetic Hindi+English document generator
│   └── prepare_dataset.py     # ingestion, dedup, quality filtering, train/val/test split
├── training/
│   ├── config.yaml
│   └── finetune_lora.py       # QLoRA fine-tuning loop
├── eval/
│   └── eval_harness.py        # F1 / exact-match / CER + regression report
├── inference/
│   ├── quantize.py            # 4-bit quantization + latency benchmark
│   └── serve_api.py           # FastAPI structured-extraction endpoint
├── retrieval/
│   └── visual_search.py       # embedding-based document search
├── scripts/
│   └── run_pipeline.sh        # runs the whole thing end to end
└── requirements.txt
```

## Notes on scope / what to customize

- Swap `Qwen/Qwen2-VL-2B-Instruct` in `training/config.yaml` for PaliGemma or InternVL2 if you
  want to compare architectures — the training loop is written against the HF `transformers`
  generic VLM interface so this should mostly be a config change plus checking the processor API.
- The synthetic generator needs a Devanagari-capable font (e.g. Noto Sans Devanagari) — path is
  a config option in `generate_synthetic.py`.
- For real (non-synthetic) data, XFUND and FUNSD are good public starting points for forms;
  swap in whatever public invoice/ID datasets you have rights to use.
