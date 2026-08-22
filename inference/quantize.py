"""
Quantizes the fine-tuned DocVL-IN checkpoint to 4-bit (bitsandbytes NF4) and benchmarks
latency/throughput/model-size against the unquantized model — this comparison table is
the actual deliverable, since "we quantized it" means nothing without before/after numbers.

Usage:
    python quantize.py --model training/output/checkpoint-final --out training/output/quantized
"""

import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig


def model_size_gb(model) -> float:
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 ** 3)


def benchmark(model, processor, n_runs: int = 20) -> dict:
    """Runs a handful of forward passes on a dummy document image + prompt and measures latency."""
    dummy_image = Image.new("RGB", (896, 1200), "white")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Extract fields as JSON."}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, images=dummy_image, return_tensors="pt").to(model.device)

    # warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=32, do_sample=False)

    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=128, do_sample=False)
        latencies.append((time.perf_counter() - start) * 1000)

    latencies.sort()
    return {
        "latency_p50_ms": latencies[len(latencies) // 2],
        "latency_p95_ms": latencies[int(len(latencies) * 0.95)],
        "model_size_gb": round(model_size_gb(model), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n-runs", type=int, default=20)
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)

    print("Loading unquantized model for baseline benchmark...")
    fp_model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    fp_model.eval()
    fp_stats = benchmark(fp_model, processor, args.n_runs)
    del fp_model
    torch.cuda.empty_cache()

    print("Loading + saving 4-bit quantized model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    q_model = AutoModelForImageTextToText.from_pretrained(
        args.model, quantization_config=bnb_config, device_map="auto"
    )
    q_model.eval()
    q_stats = benchmark(q_model, processor, args.n_runs)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    q_model.save_pretrained(args.out)
    processor.save_pretrained(args.out)

    print("\n=== Inference optimization results ===")
    print(f"{'':<20}{'FP16/BF16':<15}{'4-bit (NF4)':<15}")
    print(f"{'Model size (GB)':<20}{fp_stats['model_size_gb']:<15}{q_stats['model_size_gb']:<15}")
    print(f"{'Latency p50 (ms)':<20}{fp_stats['latency_p50_ms']:<15.1f}{q_stats['latency_p50_ms']:<15.1f}")
    print(f"{'Latency p95 (ms)':<20}{fp_stats['latency_p95_ms']:<15.1f}{q_stats['latency_p95_ms']:<15.1f}")
    print(f"\nQuantized model saved to {args.out}")
    print("Re-run eval_harness.py against this checkpoint to confirm accuracy didn't regress.")


if __name__ == "__main__":
    main()
