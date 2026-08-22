"""
Evaluation harness for DocVL-IN.

Runs a model (base or fine-tuned) over the held-out test set and computes:
  - Exact match rate (full JSON object matches ground truth exactly)
  - Field-level precision / recall / F1 (per field, and averaged)
  - Character Error Rate (CER) on text fields, for partial-credit on near-misses
    like OCR/transcription slips (useful because exact-match is harsh on names/addresses)

Writes a JSON results file plus a markdown table, and supports diffing against a
previous run for automated regression tracking across checkpoints.

Usage:
    python eval_harness.py --model Qwen/Qwen2-VL-2B-Instruct --data data/processed/test.jsonl --out eval/results_baseline.json
    python eval_harness.py --model training/output/checkpoint-final --data data/processed/test.jsonl --out eval/results_finetuned.json
    python eval_harness.py --compare eval/results_baseline.json eval/results_finetuned.json
"""

import argparse
import json
import time
from pathlib import Path

import torch
from jiwer import cer
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "data"))
from schema import prompt_for  # noqa: E402


def compute_dtype() -> torch.dtype:
    """bf16 requires Ampere+ GPUs (A10/A100/3090+); free-tier GPUs (T4, P100) don't
    support it, so detect at runtime instead of assuming."""
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


def load_model(model_path: str):
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=compute_dtype(), device_map="auto"
    )
    model.eval()
    return model, processor


def run_inference(model, processor, image_path: Path, doc_type: str) -> tuple[dict, float]:
    image = Image.open(image_path).convert("RGB")
    instruction = prompt_for(doc_type)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    latency_ms = (time.perf_counter() - start) * 1000

    text = processor.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        start_idx, end_idx = text.index("{"), text.rindex("}") + 1
        parsed = json.loads(text[start_idx:end_idx])
    except (ValueError, json.JSONDecodeError):
        parsed = {}
    return parsed, latency_ms


def field_metrics(pred: dict, gold: dict) -> dict:
    """Per-field exact match; CER for string fields where both pred and gold are non-null."""
    fields = set(gold.keys())
    tp = fp = fn = 0
    cer_scores = []
    for field in fields:
        g = gold.get(field)
        p = pred.get(field)
        if g is None and p is None:
            continue
        if g is not None and p is not None:
            if str(g).strip().lower() == str(p).strip().lower():
                tp += 1
            else:
                fp += 1
                fn += 1
                try:
                    cer_scores.append(cer(str(g), str(p)))
                except Exception:
                    pass
        elif g is not None and p is None:
            fn += 1
        elif g is None and p is not None:
            fp += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "exact_match": pred == gold,
        "avg_cer_on_mismatches": sum(cer_scores) / len(cer_scores) if cer_scores else None,
    }


def evaluate(model_path: str, data_manifest: str, image_root: str) -> dict:
    model, processor = load_model(model_path)
    rows = [json.loads(l) for l in open(data_manifest, encoding="utf-8")]

    per_example = []
    latencies = []
    for row in tqdm(rows, desc=f"evaluating {model_path}"):
        pred, latency_ms = run_inference(model, processor, Path(image_root) / row["image"], row["doc_type"])
        metrics = field_metrics(pred, row["label"])
        metrics.update({"image": row["image"], "doc_type": row["doc_type"]})
        per_example.append(metrics)
        latencies.append(latency_ms)

    latencies.sort()
    summary = {
        "model": model_path,
        "n_examples": len(rows),
        "exact_match_rate": sum(m["exact_match"] for m in per_example) / len(per_example),
        "avg_field_f1": sum(m["f1"] for m in per_example) / len(per_example),
        "avg_field_precision": sum(m["precision"] for m in per_example) / len(per_example),
        "avg_field_recall": sum(m["recall"] for m in per_example) / len(per_example),
        "latency_p50_ms": latencies[len(latencies) // 2],
        "latency_p95_ms": latencies[int(len(latencies) * 0.95)],
        "per_example": per_example,
    }
    return summary


def compare(path_a: str, path_b: str):
    a = json.load(open(path_a))
    b = json.load(open(path_b))
    print(f"\n{'metric':<25}{'A (' + a['model'] + ')':<30}{'B (' + b['model'] + ')':<30}delta")
    for key in ["exact_match_rate", "avg_field_f1", "avg_field_precision", "avg_field_recall", "latency_p50_ms"]:
        delta = b[key] - a[key]
        print(f"{key:<25}{a[key]:<30.4f}{b[key]:<30.4f}{delta:+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str)
    ap.add_argument("--data", type=str)
    ap.add_argument("--image-root", type=str, default="data/processed")
    ap.add_argument("--out", type=str)
    ap.add_argument("--compare", nargs=2, metavar=("RESULTS_A", "RESULTS_B"))
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    summary = evaluate(args.model, args.data, args.image_root)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2, ensure_ascii=False)

    print(f"\nExact match: {summary['exact_match_rate']:.3f}")
    print(f"Field F1:    {summary['avg_field_f1']:.3f}")
    print(f"Latency p50: {summary['latency_p50_ms']:.1f} ms")
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
