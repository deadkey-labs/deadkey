"""Reusable RTX 5090 QLoRA DPO Trainer + Standalone Merged GGUF Builder.

Trains a zero-PII (or cartridge-specific) DPO adapter via Unsloth + TRL,
merges cleanly to 16-bit HF, converts to GGUF via llama-b12780's converter,
quantizes to Q4_K_M / Q5_K_M (ZERO runtime LoRA speed tax!), and cleans up
large BF16 intermediates automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", r"D:\Models\HF_Cache")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
LLAMA_CONVERT_PKG = ROOT / "engines" / "llama-b12780-cu135" / "llama_convert_pkg"
QUANTIZE_EXE = ROOT / "engines" / "llama-b12780-cu135" / "llama-quantize.exe"


def main() -> int:
    ap = argparse.ArgumentParser(description="RTX 5090 QLoRA DPO -> Standalone Merged GGUF")
    ap.add_argument("--model", required=True, help="Unsloth 4-bit HF repo ID (e.g. unsloth/Qwen3-8B-unsloth-bnb-4bit)")
    ap.add_argument("--dataset", required=True, help="JSONL file with {prompt, chosen, rejected}")
    ap.add_argument("--adapter-dir", required=True, help="Directory to save PEFT LoRA adapter")
    ap.add_argument("--out-gguf", required=True, help="Final quantized GGUF output path")
    ap.add_argument("--quant", default="Q4_K_M", help="GGUF quant type (Q4_K_M or Q5_K_M)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=8e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=1536)
    ap.add_argument("--blank-mind", action="store_true", help="Zero-PII engineering system prompt")
    ap.add_argument("--no-merge", action="store_true", help="Only save PEFT LoRA + GGUF LoRA")
    ap.add_argument("--cleanup", action="store_true", help="Delete temporary BF16 merged dir & BF16 GGUF")
    args = ap.parse_args()

    ds_path = Path(args.dataset)
    rows = [json.loads(ln) for ln in ds_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    sha = hashlib.sha256(ds_path.read_bytes()).hexdigest()[:12]
    print(f"[DPO-GGUF] Model={args.model} | Pairs={len(rows)} (sha {sha}) | Epochs={args.epochs}", flush=True)

    t0 = time.perf_counter()
    from unsloth import FastLanguageModel, PatchDPOTrainer
    import torch
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    print(f"[DPO-GGUF] GPU={torch.cuda.get_device_name(0)} | bf16={torch.cuda.is_bf16_supported()}", flush=True)
    PatchDPOTrainer()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.rank,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    if args.blank_mind:
        sys_msg = (
            "You are a concise, accurate engineering assistant. "
            "Answer directly using only verified facts and tool receipts provided in context. "
            "Never guess file sizes, line counts, file contents, or claim/confirm actions you did not execute."
        )
    else:
        sys_msg = (
            "You are deadkey, the operator's local companion on Colossus (Deadkey). "
            "Speak directly in warm, grounded first-person voice. "
            "Follow Disk-Equals-Truth strictly: never guess file sizes, hashes, or claim unexecuted actions."
        )

    def _fmt(r: dict) -> dict:
        msgs = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": str(r["prompt"]).strip()},
        ]
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            try:
                prompt_text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            prompt_text = f"System: {sys_msg}\nUser: {r['prompt']}\nAssistant: "
        end_tok = tokenizer.eos_token or ""
        return {
            "prompt": prompt_text,
            "chosen": str(r["chosen"]).strip() + end_tok,
            "rejected": str(r["rejected"]).strip() + end_tok,
        }

    ds = Dataset.from_list([_fmt(r) for r in rows])
    adapter_dir = Path(args.adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    cfg = DPOConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        learning_rate=args.lr,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        beta=args.beta,
        max_length=args.max_seq_len,
        max_prompt_length=512,
        report_to="none",
        save_strategy="no",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    train_out = trainer.train()
    train_s = round(time.perf_counter() - t0, 1)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[DPO-GGUF] Train DONE in {train_s}s | loss={train_out.training_loss:.4f} -> {adapter_dir}", flush=True)

    if args.no_merge:
        return 0

    out_gguf = Path(args.out_gguf)
    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    stem = out_gguf.stem
    merged_dir = Path(r"D:\Models\HF_Cache") / f"{stem}_merged_16bit"
    bf16_gguf = Path(r"D:\Models\HF_Cache") / f"{stem}-BF16.gguf"

    print(f"[DPO-GGUF] Merging 16-bit -> {merged_dir}...", flush=True)
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    print(f"[DPO-GGUF] Converting HF -> BF16 GGUF ({bf16_gguf})...", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{LLAMA_CONVERT_PKG}{os.pathsep}{LLAMA_CONVERT_PKG / 'gguf-py'}"
    r1 = subprocess.run(
        [sys.executable, str(LLAMA_CONVERT_PKG / "convert_hf_to_gguf.py"),
         str(merged_dir), "--outfile", str(bf16_gguf), "--outtype", "bf16"],
        env=env,
    )
    if r1.returncode != 0:
        raise RuntimeError(f"convert_hf_to_gguf.py failed rc={r1.returncode}")

    print(f"[DPO-GGUF] Quantizing -> {out_gguf} ({args.quant})...", flush=True)
    r2 = subprocess.run([str(QUANTIZE_EXE), str(bf16_gguf), str(out_gguf), args.quant])
    if r2.returncode != 0:
        raise RuntimeError(f"llama-quantize.exe failed rc={r2.returncode}")

    sz_gb = out_gguf.stat().st_size / (1024 ** 3)
    if args.cleanup:
        shutil.rmtree(merged_dir, ignore_errors=True)
        bf16_gguf.unlink(missing_ok=True)
        print("[DPO-GGUF] Cleaned temporary BF16 intermediates.", flush=True)

    total_s = round(time.perf_counter() - t0, 1)
    meta = {
        "base_model": args.model,
        "dataset": str(ds_path),
        "dataset_sha": sha,
        "pairs": len(rows),
        "epochs": args.epochs,
        "train_loss": round(float(train_out.training_loss), 4),
        "train_s": train_s,
        "total_s": total_s,
        "out_gguf": str(out_gguf),
        "out_gib": round(sz_gb, 2),
        "quant": args.quant,
    }
    (adapter_dir / "train_merge_receipt.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[DPO-GGUF] ALL DONE in {total_s}s -> {out_gguf} ({sz_gb:.2f} GiB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
