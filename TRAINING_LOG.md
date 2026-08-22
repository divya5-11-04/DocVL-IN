# Training Log

Keep this updated as you actually run the pipeline. A recruiter reading "OOM at batch
size 4 on a T4, fixed by dropping to batch size 2 + grad accumulation 16" is more
convincing than any accuracy number — it's proof you ran real training, not just code
that looks plausible.

## Template per entry

### YYYY-MM-DD — <short description>
- **What I ran:** (command / config)
- **What broke:** (error message, symptom)
- **Root cause:**
- **Fix:**
- **Result after fix:**

---

### Example (delete once you have real entries)

### 2026-XX-XX — First fine-tuning attempt, OOM
- **What I ran:** `python training/finetune_lora.py --config training/config.yaml` on a Colab T4 (15GB)
- **What broke:** `CUDA out of memory` at step 3
- **Root cause:** per_device_train_batch_size=2 with max_image_size=896 was too large combined with 4-bit base + LoRA adapters + optimizer states
- **Fix:** dropped max_image_size to 672, batch size to 1, grad_accumulation to 16 (same effective batch size)
- **Result after fix:** training ran to completion, ~2.5 hrs for 3 epochs on 450 examples
