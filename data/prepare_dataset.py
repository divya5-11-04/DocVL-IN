"""
Builds the final train/val/test manifests for DocVL-IN by:
  1. Ingesting synthetic data (from generate_synthetic.py) and any public datasets
     placed in the same manifest.jsonl format.
  2. Deduplicating near-identical images via perceptual hashing.
  3. Filtering low-quality images (too blurry, too small, unreadable).
  4. Splitting into train/val/test and writing final .jsonl files consumed by
     training/finetune_lora.py and eval/eval_harness.py.

Expected input manifest format (one JSON object per line):
    {"image": "images/foo.png", "doc_type": "invoice", "label": {...}}

Usage:
    python prepare_dataset.py --synthetic data/synthetic --out data/processed
    python prepare_dataset.py --synthetic data/synthetic --extra data/public_xfund --out data/processed
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm


def blur_score(image_path: Path) -> float:
    """Variance of the Laplacian — a standard, cheap blur/quality metric. Higher = sharper."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def load_manifest(source_dir: Path) -> list[dict]:
    manifest_path = source_dir / "manifest.jsonl"
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            row["_source_dir"] = str(source_dir)
            rows.append(row)
    return rows


def dedup(rows: list[dict], threshold: int = 4) -> list[dict]:
    """Drops near-duplicate images using perceptual hashing (phash). threshold = max
    Hamming distance to still count as a duplicate; smaller = stricter."""
    seen_hashes = []
    kept = []
    for row in tqdm(rows, desc="dedup"):
        img_path = Path(row["_source_dir"]) / row["image"]
        try:
            h = imagehash.phash(Image.open(img_path))
        except Exception:
            continue
        is_dup = any(h - prev <= threshold for prev in seen_hashes)
        if not is_dup:
            seen_hashes.append(h)
            kept.append(row)
    print(f"dedup: {len(rows)} -> {len(kept)}")
    return kept


def quality_filter(rows: list[dict], min_blur: float = 15.0, min_size: int = 200) -> list[dict]:
    kept = []
    for row in tqdm(rows, desc="quality filter"):
        img_path = Path(row["_source_dir"]) / row["image"]
        try:
            with Image.open(img_path) as im:
                w, h = im.size
            if w < min_size or h < min_size:
                continue
            if blur_score(img_path) < min_blur:
                continue
            kept.append(row)
        except Exception:
            continue
    print(f"quality filter: {len(rows)} -> {len(kept)}")
    return kept


def split(rows: list[dict], val_frac=0.1, test_frac=0.1, seed=42) -> dict[str, list[dict]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))
    n_val = int(len(rows) * val_frac)
    n_test = int(len(rows) * test_frac)
    val_idx = set(idx[:n_val])
    test_idx = set(idx[n_val:n_val + n_test])
    train, val, test = [], [], []
    for i, row in enumerate(rows):
        (val if i in val_idx else test if i in test_idx else train).append(row)
    return {"train": train, "val": val, "test": test}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=str, required=True)
    ap.add_argument("--extra", type=str, nargs="*", default=[],
                     help="additional dataset dirs in the same manifest.jsonl format")
    ap.add_argument("--out", type=str, default="data/processed")
    args = ap.parse_args()

    out_dir = Path(args.out)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    all_rows = load_manifest(Path(args.synthetic))
    for extra_dir in args.extra:
        all_rows += load_manifest(Path(extra_dir))
    print(f"loaded {len(all_rows)} total rows from {1 + len(args.extra)} source(s)")

    all_rows = dedup(all_rows)
    all_rows = quality_filter(all_rows)

    # copy images into a unified location and rewrite paths relative to out_dir
    for i, row in enumerate(all_rows):
        src = Path(row["_source_dir"]) / row["image"]
        dst_name = f"{i:06d}_{Path(row['image']).name}"
        shutil.copy(src, out_dir / "images" / dst_name)
        row["image"] = f"images/{dst_name}"
        del row["_source_dir"]

    splits = split(all_rows)
    for name, rows in splits.items():
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} examples -> {out_dir / (name + '.jsonl')}")


if __name__ == "__main__":
    main()
