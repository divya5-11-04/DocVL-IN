"""
Embedding-based visual search over a folder of documents — the "retrieval-augmented
workflows" / "visual search" use case from the JD. Given a folder of scanned documents,
builds a vector index over their image embeddings (using the VLM's own vision encoder,
so no separate embedding model is needed) and lets you query with a new image ("find
documents similar to this one") or a text query ("find invoices from vendor X").

Usage:
    python visual_search.py build --docs data/processed/images --index retrieval/index.pkl
    python visual_search.py query --index retrieval/index.pkl --image samples/query.jpg --top-k 5
    python visual_search.py query --index retrieval/index.pkl --text "invoice from Tata" --top-k 5
"""

import argparse
import pickle
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

# A dedicated CLIP-style model is used for embeddings rather than the fine-tuned VLM
# itself: retrieval wants a good *similarity* space, which CLIP is trained for
# directly, whereas the VLM was fine-tuned for generation, not embedding quality.
EMBED_MODEL = "openai/clip-vit-base-patch32"


class DocIndex:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(EMBED_MODEL)
        self.model = AutoModel.from_pretrained(EMBED_MODEL)
        self.model.eval()
        self.paths: list[str] = []
        self.embeddings = None  # (N, D) numpy array after build()

    @torch.no_grad()
    def embed_image(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt")
        feats = self.model.get_image_features(**inputs)
        return (feats / feats.norm(dim=-1, keepdim=True)).squeeze(0).numpy()

    @torch.no_grad()
    def embed_text(self, text: str):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        feats = self.model.get_text_features(**inputs)
        return (feats / feats.norm(dim=-1, keepdim=True)).squeeze(0).numpy()

    def build(self, docs_dir: Path):
        import numpy as np
        image_paths = sorted([p for p in docs_dir.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
        embeddings = []
        for p in tqdm(image_paths, desc="embedding documents"):
            embeddings.append(self.embed_image(Image.open(p).convert("RGB")))
        self.paths = [str(p) for p in image_paths]
        self.embeddings = np.stack(embeddings)

    def search(self, query_embedding, top_k: int = 5):
        sims = cosine_similarity(query_embedding.reshape(1, -1), self.embeddings)[0]
        top_idx = sims.argsort()[::-1][:top_k]
        return [(self.paths[i], float(sims[i])) for i in top_idx]

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump({"paths": self.paths, "embeddings": self.embeddings}, f)

    @classmethod
    def load(cls, path: Path):
        idx = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx.paths, idx.embeddings = data["paths"], data["embeddings"]
        return idx


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    build_p = sub.add_parser("build")
    build_p.add_argument("--docs", type=str, required=True)
    build_p.add_argument("--index", type=str, required=True)

    query_p = sub.add_parser("query")
    query_p.add_argument("--index", type=str, required=True)
    query_p.add_argument("--image", type=str)
    query_p.add_argument("--text", type=str)
    query_p.add_argument("--top-k", type=int, default=5)

    args = ap.parse_args()

    if args.cmd == "build":
        idx = DocIndex()
        idx.build(Path(args.docs))
        idx.save(Path(args.index))
        print(f"Indexed {len(idx.paths)} documents -> {args.index}")

    elif args.cmd == "query":
        idx = DocIndex.load(Path(args.index))
        if args.image:
            q_emb = idx.embed_image(Image.open(args.image).convert("RGB"))
        elif args.text:
            q_emb = idx.embed_text(args.text)
        else:
            raise SystemExit("Provide --image or --text for the query.")

        for path, score in idx.search(q_emb, args.top_k):
            print(f"{score:.4f}  {path}")


if __name__ == "__main__":
    main()
