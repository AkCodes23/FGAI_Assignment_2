"""Decoder-only adaptation: instruction prompting and QLoRA fine-tuning for LLaMA / Gemma families."""
from __future__ import annotations

import gc
import json
import hashlib
import math
from huggingface_hub import snapshot_download
from safetensors import safe_open
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM = (
    "You are a careful professional translator. Translate English sentences into natural Hindi. "
    "Preserve meaning, names and numbers. Never add explanations, transliteration, notes or quotes."
)
INSTRUCTION = "Translate the following English sentence into Hindi. Reply with the Hindi translation only."


def user_text(src: str) -> str:
    return f"{INSTRUCTION}\n\nEnglish: {src}\nHindi:"


def build_messages(src: str, shots: Sequence[dict[str, str]] = ()) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM}]
    for shot in shots:
        messages.append({"role": "user", "content": user_text(shot["en"])})
        messages.append({"role": "assistant", "content": shot["hi"]})
    messages.append({"role": "user", "content": user_text(src)})
    return messages


def render_prompt(tokenizer, src: str, shots: Sequence[dict[str, str]] = ()) -> str:
    return tokenizer.apply_chat_template(build_messages(src, shots), tokenize=False, add_generation_prompt=True)


def prompt_ids(tokenizer, prompt: str) -> list[int]:
    """Tokenise a rendered chat prompt without adding a second BOS.

    Gemma's template already starts with ``<bos>`` while TinyLlama's does not, so a
    single ``add_special_tokens`` value would either duplicate the marker or drop it.
    Training and generation both call this helper, keeping the prompt format identical.
    """
    bos = tokenizer.bos_token
    add_special = not (bos and prompt.startswith(bos))
    return tokenizer(prompt, add_special_tokens=add_special)["input_ids"]


def load_tokenizer(model_id: str, revision: str = "main"):
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def count_full_checkpoint_parameters(model_id: str, revision: str = "main") -> int:
    """Count original checkpoint weights from cached safetensors headers, before quantization.

    This reads tensor shapes only. It does not load weights or access the network.
    The decoder models in this assignment use untied embeddings, so counting each
    stored tensor shape once gives the full unquantized parameter count.
    """
    snapshot = Path(snapshot_download(model_id, revision=revision, local_files_only=True))
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        shard_names = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
        weight_files = [snapshot / name for name in shard_names]
    else:
        weight_files = sorted(snapshot.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No cached safetensors checkpoint found for {model_id}@{revision}")
    total = 0
    for weight_file in weight_files:
        with safe_open(str(weight_file), framework="pt", device="cpu") as tensors:
            for key in tensors.keys():
                total += math.prod(tensors.get_slice(key).get_shape())
    return total


def load_model(model_id: str, revision: str = "main", load_in_4bit: bool = True, device: str = "cuda"):
    kwargs: dict[str, Any] = {"revision": revision, "torch_dtype": torch.bfloat16, "device_map": {"": device}}
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model


@torch.no_grad()
def generate_texts(
    model,
    tokenizer,
    prompts: Sequence[str],
    max_new_tokens: int = 80,
    batch_size: int = 8,
    temperature: float | None = None,
    top_p: float | None = None,
    max_input_tokens: int = 448,
) -> dict[str, Any]:
    """Deterministic when temperature is None/0. Returns texts, token counts and generation-only timing."""
    tokenizer.padding_side = "left"
    texts: list[str] = []
    new_tokens = 0
    hit_cap = 0
    eos_ids = getattr(model.generation_config, "eos_token_id", None) or tokenizer.eos_token_id
    if not isinstance(eos_ids, (list, tuple)):
        eos_ids = [eos_ids]
    eos_ids = [int(x) for x in eos_ids if x is not None]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for start in range(0, len(prompts), batch_size):
        batch = list(prompts[start : start + batch_size])
        add_special = not (tokenizer.bos_token and batch[0].startswith(tokenizer.bos_token))
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_input_tokens,
                        add_special_tokens=add_special).to(model.device)
        sampling = temperature is not None and temperature > 0
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=sampling,
            temperature=temperature if sampling else None,
            top_p=top_p if sampling else None,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids,
        )
        gen = out[:, enc["input_ids"].shape[1] :]
        new_tokens += int((gen != tokenizer.pad_token_id).sum())
        for row in gen.tolist():
            if tokenizer.pad_token_id in eos_ids:
                stopped = any(t in eos_ids for t in row)
            else:
                toks = [t for t in row if t != tokenizer.pad_token_id]
                stopped = bool(toks) and toks[-1] in eos_ids
            if not stopped:
                hit_cap += 1
        texts.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    seconds = time.perf_counter() - t0
    return {
        "texts": [clean_output(t) for t in texts],
        "seconds": round(seconds, 3),
        "new_tokens": new_tokens,
        "hit_max_new_tokens": hit_cap,
        "tokens_per_second": round(new_tokens / max(seconds, 1e-9), 2),
        "sentences_per_second": round(len(prompts) / max(seconds, 1e-9), 2),
    }


def clean_output(text: str) -> str:
    for marker in ("<end_of_turn>", "</s>", "<|eot_id|>", "<|endoftext|>", "<|user|>", "<|system|>", "<|assistant|>"):
        text = text.replace(marker, "\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    kept: list[str] = []
    for line in lines:
        low = line.lower()
        if low.startswith("english:"):
            break
        if not kept and (low.startswith("translate the following") or low.startswith("hindi translation:")):
            if low.startswith("hindi translation:") and len(line) > 18:
                kept.append(line.split(":", 1)[1].strip())
            continue
        kept.append(line)
    text = " ".join(kept).strip()
    for prefix in ("Hindi:", "Hindi translation:", "Translation:", "**Hindi:**"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return text.strip().strip('"').strip("'")


# --------------------------------------------------------------------------
# QLoRA fine-tuning
# --------------------------------------------------------------------------
def _training_example(tokenizer, src: str, tgt: str, max_len: int) -> dict[str, Any]:
    prompt = render_prompt(tokenizer, src, shots=())
    prompt_ids_ = prompt_ids(tokenizer, prompt)
    target_ids = tokenizer(tgt, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    truncated = len(prompt_ids_) + len(target_ids) > max_len
    input_ids = (prompt_ids_ + target_ids)[:max_len]
    labels = ([-100] * len(prompt_ids_) + target_ids)[:max_len]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids),
            "truncated": truncated}


def _collate(batch: list[dict[str, Any]], pad_id: int):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, row in enumerate(batch):
        n = len(row["input_ids"])
        input_ids[i, :n] = torch.tensor(row["input_ids"])
        labels[i, :n] = torch.tensor(row["labels"])
        mask[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": mask}


def train_qlora(
    model_id: str,
    train_rows: Sequence[dict[str, str]],
    val_rows: Sequence[dict[str, str]],
    out_dir: Path,
    revision: str = "main",
    load_in_4bit: bool = True,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    lr: float = 2e-4,
    epochs: int = 2,
    batch_size: int = 4,
    grad_accum: int = 4,
    max_seq_len: int = 384,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.01,
    seed: int = 42,
    device: str = "cuda",
    logger=None,
    jsonl=None,
    smoke: bool = False,
) -> dict[str, Any]:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    tokenizer = load_tokenizer(model_id, revision)
    tokenizer.padding_side = "right"
    model = load_model(model_id, revision, load_in_4bit=load_in_4bit, device=device)
    model.config.use_cache = False
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if logger:
        logger.info("LoRA trainable params: %s / %s (%.2f%%)", f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    if smoke:
        train_rows, val_rows, epochs = train_rows[:80], val_rows[:8], 1

    train_examples = [_training_example(tokenizer, row["en"], row["hi"], max_seq_len) for row in train_rows]
    val_examples = [_training_example(tokenizer, row["en"], row["hi"], max_seq_len) for row in val_rows]

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
    )
    steps_per_epoch = math.ceil(len(train_examples) / (batch_size * grad_accum))
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    pad_id = tokenizer.pad_token_id
    idx_order = list(range(len(train_examples)))
    rng = torch.Generator().manual_seed(seed)
    history = []
    t_start = time.perf_counter()
    global_step = 0
    model.train()

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(idx_order), generator=rng).tolist()
        running, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        t_epoch = time.perf_counter()
        for step, start in enumerate(range(0, len(perm), batch_size)):
            ids = perm[start : start + batch_size]
            batch = _collate([train_examples[i] for i in ids], pad_id)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()
            running += float(out.loss.detach()) * len(ids)
            seen += len(ids)
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if jsonl and global_step % 20 == 0:
                    jsonl.log(event="qlora_step", model=model_id, epoch=epoch, step=global_step,
                              loss=round(float(out.loss.detach()), 4), lr=round(optimizer.param_groups[0]["lr"], 8))
        val_loss = evaluate_lm_loss(model, val_examples, batch_size, pad_id)
        row = {"epoch": epoch, "train_loss": round(running / max(1, seen), 4), "val_loss": round(val_loss, 4),
               "seconds": round(time.perf_counter() - t_epoch, 1), "step": global_step}
        history.append(row)
        if logger:
            logger.info("epoch %d | train %.4f | val %.4f | %.1fs", epoch, row["train_loss"], val_loss, row["seconds"])
        model.save_pretrained(out_dir / f"checkpoint-epoch{epoch}")
    train_seconds = time.perf_counter() - t_start
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    peak_vram = round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0.0

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    summary = {
        "model_id": model_id,
        "revision": revision,
        "load_in_4bit": load_in_4bit,
        "lora_r": r,
        "lora_alpha": alpha,
        "trainable_params": trainable,
        "total_params": total,
        "train_seconds": round(train_seconds, 1),
        "train_rows": len(train_rows),
        "train_subset_sha256": hashlib.sha256(
            "\n".join(f"{row['en']}\t{row['hi']}" for row in train_rows).encode("utf-8")
        ).hexdigest(),
        "truncated_train_rows": sum(1 for ex in train_examples if ex.get("truncated")),
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "max_seq_len": max_seq_len,
        "lr": lr,
        "seed": seed,
        "peak_vram_gb": peak_vram,
        "history": history,
        "output_dir": str(out_dir),
    }
    return summary


@torch.no_grad()
def evaluate_lm_loss(model, examples, batch_size, pad_id) -> float:
    model.eval()
    total, tokens = 0.0, 0
    for start in range(0, len(examples), batch_size):
        batch = _collate(examples[start : start + batch_size], pad_id)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out = model(**batch)
        n_tok = int((batch["labels"] != -100).sum())
        total += float(out.loss.detach()) * n_tok
        tokens += n_tok
    model.train()
    return total / max(1, tokens)
