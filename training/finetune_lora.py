"""
QLoRA fine-tuning of a vision-language model on the DocVL-IN structured-extraction task.

The model is trained to take (document image, extraction prompt) -> JSON string matching
the schema in data/schema.py. This keeps the task identical to inference: no separate
classification/regression heads, just next-token prediction on the target JSON, which is
what lets a single VLM checkpoint serve the whole "document processing" use case Sarvam
describes (forms, invoices, structured output extraction).

Usage:
    python finetune_lora.py --config config.yaml

Expect this to break the first few times — OOM on batch size, tokenizer/processor
chat-template mismatches, and image-resizing edge cases are the most common failure
modes with VLM fine-tuning. Log what broke and how you fixed it in TRAINING_LOG.md;
that debugging trail is part of the point of this project, not an inconvenience.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "data"))
from schema import prompt_for  # noqa: E402


class DocVLDataset(Dataset):
    """Reads a manifest.jsonl of {image, doc_type, label} and formats each example as a
    chat-style (image + instruction) -> JSON-string training pair."""

    def __init__(self, manifest_path: str, image_root: str, processor, max_image_size: int):
        self.rows = [json.loads(l) for l in open(manifest_path, encoding="utf-8")]
        self.image_root = Path(image_root)
        self.processor = processor
        self.max_image_size = max_image_size

    def __len__(self):
        return len(self.rows)

    def _load_image(self, rel_path: str) -> Image.Image:
        img = Image.open(self.image_root / rel_path).convert("RGB")
        img.thumbnail((self.max_image_size, self.max_image_size))
        return img

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = self._load_image(row["image"])
        instruction = prompt_for(row["doc_type"])
        target_json = json.dumps(row["label"], ensure_ascii=False)

        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]},
            {"role": "assistant", "content": [{"type": "text", "text": target_json}]},
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False)
        model_inputs = self.processor(text=prompt, images=image, return_tensors="pt", padding=False)

        # labels = input_ids, with the prompt portion masked out so loss is computed
        # only on the JSON completion, not the instruction text.
        labels = model_inputs["input_ids"].clone()
        prompt_only = self.processor.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
        prompt_only_ids = self.processor(text=prompt_only, images=image, return_tensors="pt")["input_ids"]
        prompt_len = prompt_only_ids.shape[1]
        labels[:, :prompt_len] = -100

        item = {k: v.squeeze(0) for k, v in model_inputs.items()}
        item["labels"] = labels.squeeze(0)
        return item


def collate_fn(batch, pad_token_id):
    """Pads variable-length sequences in a batch. Image tensors from most HF VLM
    processors are already fixed-size per model, so only text fields need padding."""
    max_len = max(x["input_ids"].shape[0] for x in batch)
    out = {}
    for key in batch[0]:
        if key == "pixel_values" or "image" in key:
            out[key] = torch.stack([x[key] for x in batch])
            continue
        pad_val = pad_token_id if key == "input_ids" else (-100 if key == "labels" else 0)
        padded = []
        for x in batch:
            t = x[key]
            pad_len = max_len - t.shape[0]
            if pad_len > 0:
                t = torch.cat([t, torch.full((pad_len,), pad_val, dtype=t.dtype)])
            padded.append(t)
        out[key] = torch.stack(padded)
    return out


def resolve_path(root: Path, p: str) -> str:
    """Relative paths in config.yaml are relative to the repo root (the config file's
    parent directory), never to the process's cwd — cwd is unreliable across notebook
    cells, restarted sessions, and different invocation locations."""
    p_path = Path(p)
    return str(p_path if p_path.is_absolute() else root / p_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()
    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(config_path))
    repo_root = config_path.parent.parent  # config.yaml lives in training/, repo root is one level up

    cfg["data"]["train_manifest"] = resolve_path(repo_root, cfg["data"]["train_manifest"])
    cfg["data"]["val_manifest"] = resolve_path(repo_root, cfg["data"]["val_manifest"])
    cfg["data"]["image_root"] = resolve_path(repo_root, cfg["data"]["image_root"])
    cfg["training"]["output_dir"] = resolve_path(repo_root, cfg["training"]["output_dir"])

    # Never let this block waiting on an interactive prompt (e.g. running as a subprocess
    # cell in a notebook). W&B is opt-in only, and only actually used if a key is present.
    if not cfg["logging"]["use_wandb"] or not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_MODE"] = "disabled"
        cfg["logging"]["use_wandb"] = False

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["model"]["load_in_4bit"],
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    processor = AutoProcessor.from_pretrained(cfg["model"]["base_model"])
    model = AutoModelForImageTextToText.from_pretrained(
        cfg["model"]["base_model"],
        quantization_config=bnb_config if cfg["model"]["load_in_4bit"] else None,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    if cfg["model"]["load_in_4bit"]:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = DocVLDataset(cfg["data"]["train_manifest"], cfg["data"]["image_root"],
                             processor, cfg["data"]["max_image_size"])
    val_ds = DocVLDataset(cfg["data"]["val_manifest"], cfg["data"]["image_root"],
                           processor, cfg["data"]["max_image_size"])

    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    training_args = TrainingArguments(
        output_dir=cfg["training"]["output_dir"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        logging_steps=cfg["training"]["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["training"]["eval_steps"],
        save_steps=cfg["training"]["save_steps"],
        save_total_limit=cfg["training"]["save_total_limit"],
        bf16=cfg["training"]["bf16"],
        max_grad_norm=cfg["training"]["max_grad_norm"],
        seed=cfg["training"]["seed"],
        report_to=["wandb"] if cfg["logging"]["use_wandb"] else [],
        run_name=cfg["logging"]["project_name"],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda batch: collate_fn(batch, pad_id),
    )

    trainer.train()
    trainer.save_model(str(Path(cfg["training"]["output_dir"]) / "checkpoint-final"))
    processor.save_pretrained(str(Path(cfg["training"]["output_dir"]) / "checkpoint-final"))
    print("Training complete. Checkpoint saved to", cfg["training"]["output_dir"] + "/checkpoint-final")


if __name__ == "__main__":
    main()
