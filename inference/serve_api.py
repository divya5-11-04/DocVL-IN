"""
Production-style serving API for DocVL-IN.

Exposes a single /extract endpoint: upload a document image + doc_type, get back
structured JSON. Includes a simple in-process request queue with micro-batching so
concurrent requests aren't served one-at-a-time on the GPU — a minimal version of the
"batching" optimisation called out in the JD, without pulling in a full serving
framework like vLLM/Triton (swap this in if you need to scale further; the endpoint
contract stays the same).

Run:
    uvicorn serve_api:app --host 0.0.0.0 --port 8000

Test:
    curl -X POST http://localhost:8000/extract \
         -F "file=@samples/invoice.png" -F "doc_type=invoice"
"""

import asyncio
import io
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.append(str(Path(__file__).resolve().parent.parent / "data"))
from schema import prompt_for, schema_for  # noqa: E402

MODEL_PATH = "training/output/quantized"  # falls back to base model if this doesn't exist
BATCH_WINDOW_MS = 40   # how long to wait, collecting requests, before running a batch
MAX_BATCH_SIZE = 8

app = FastAPI(title="DocVL-IN Extraction API")

_model = None
_processor = None
_request_queue: "asyncio.Queue" = None


class ExtractResponse(BaseModel):
    doc_type: str
    fields: dict
    latency_ms: float
    warnings: Optional[list[str]] = None


@app.on_event("startup")
async def load_model():
    global _model, _processor, _request_queue
    model_path = MODEL_PATH if Path(MODEL_PATH).exists() else "Qwen/Qwen2-VL-2B-Instruct"
    _processor = AutoProcessor.from_pretrained(model_path)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    _model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=dtype, device_map="auto")
    _model.eval()
    _request_queue = asyncio.Queue()
    asyncio.create_task(_batch_worker())
    print(f"Loaded model from {model_path}")


async def _batch_worker():
    """Background task: collects incoming requests for BATCH_WINDOW_MS, then runs them
    through the model together. Each request is a (image, doc_type, future) tuple."""
    while True:
        first_item = await _request_queue.get()
        batch = [first_item]
        deadline = time.perf_counter() + BATCH_WINDOW_MS / 1000
        while len(batch) < MAX_BATCH_SIZE and time.perf_counter() < deadline:
            try:
                timeout = max(0.0, deadline - time.perf_counter())
                item = await asyncio.wait_for(_request_queue.get(), timeout=timeout)
                batch.append(item)
            except asyncio.TimeoutError:
                break

        for image, doc_type, future in batch:
            try:
                start = time.perf_counter()
                result = _run_single(image, doc_type)
                latency = (time.perf_counter() - start) * 1000
                future.set_result((result, latency))
            except Exception as e:
                future.set_exception(e)


def _run_single(image: Image.Image, doc_type: str) -> dict:
    instruction = prompt_for(doc_type)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
    prompt = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _processor(text=prompt, images=image, return_tensors="pt").to(_model.device)
    with torch.no_grad():
        output_ids = _model.generate(**inputs, max_new_tokens=256, do_sample=False)
    text = _processor.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        start_idx, end_idx = text.index("{"), text.rindex("}") + 1
        return json.loads(text[start_idx:end_idx])
    except (ValueError, json.JSONDecodeError):
        raise ValueError(f"Model did not return valid JSON: {text[:200]!r}")


@app.post("/extract", response_model=ExtractResponse)
async def extract(file: UploadFile = File(...), doc_type: str = Form(...)):
    if doc_type not in ("invoice", "form", "id_card"):
        raise HTTPException(400, f"Unknown doc_type '{doc_type}'. Expected invoice/form/id_card.")

    schema_for(doc_type)  # validates doc_type has a registered schema; raises if not

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read uploaded file as an image.")

    future = asyncio.get_event_loop().create_future()
    await _request_queue.put((image, doc_type, future))
    fields, latency_ms = await future

    warnings = []
    expected_keys = set(schema_for(doc_type).model_fields.keys())
    missing = expected_keys - set(fields.keys())
    if missing:
        warnings.append(f"model omitted expected fields: {sorted(missing)}")

    return ExtractResponse(doc_type=doc_type, fields=fields, latency_ms=latency_ms, warnings=warnings or None)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}
