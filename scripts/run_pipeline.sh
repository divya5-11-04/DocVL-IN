#!/usr/bin/env bash
# Runs the full DocVL-IN pipeline end to end. Run from the repo root:
#   bash scripts/run_pipeline.sh
set -euo pipefail

echo "=== 1/6: Generating synthetic data ==="
python data/generate_synthetic.py --n 500 --out data/synthetic

echo "=== 2/6: Preparing dataset (dedup + quality filter + split) ==="
python data/prepare_dataset.py --synthetic data/synthetic --out data/processed

echo "=== 3/6: Baseline evaluation (zero-shot) ==="
python eval/eval_harness.py \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --data data/processed/test.jsonl \
    --out eval/results_baseline.json

echo "=== 4/6: Fine-tuning ==="
python training/finetune_lora.py --config training/config.yaml

echo "=== 5/6: Evaluating fine-tuned model ==="
python eval/eval_harness.py \
    --model training/output/checkpoint-final \
    --data data/processed/test.jsonl \
    --out eval/results_finetuned.json

python eval/eval_harness.py --compare eval/results_baseline.json eval/results_finetuned.json

echo "=== 6/6: Quantizing + benchmarking inference ==="
python inference/quantize.py --model training/output/checkpoint-final --out training/output/quantized

echo "Pipeline complete. Start the API with:"
echo "  uvicorn inference.serve_api:app --host 0.0.0.0 --port 8000"
