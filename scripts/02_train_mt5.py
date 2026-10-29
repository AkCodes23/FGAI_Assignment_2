#!/usr/bin/env python
"""Fine-tune mT5 (encoder-decoder) on the shared English-Hindi training split."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: E402

from fgai_mt.config import dump_config, load_config, project_path  # noqa: E402
from fgai_mt.data import load_split  # noqa: E402
from fgai_mt.metrics import corpus_scores, output_diagnostics  # noqa: E402
from fgai_mt.utils import JsonlLogger, get_logger, set_seed, write_json, write_jsonl  # noqa: E402

RESULTS = project_path("results")
TRANSLATIONS = RESULTS / "translations"
OUT_DIR = project_path("models", "mt5_base")


def tokenize_rows(rows, tok, prefix, max_src, max_tgt):
    src = [prefix + r["en"] for r in rows]
    enc = tok(src, max_length=max_src, truncation=True, padding=False)
    with tok.as_target_tokenizer() if hasattr(tok, "as_target_tokenizer") else _nullcontext():
        lab = tok([r["hi"] for r in rows], max_length=max_tgt, truncation=True, padding=False)
    return [
        {"input_ids": list(a), "attention_mask": list(b), "labels": list(c)}
        for a, b, c in zip(enc["input_ids"], enc["attention_mask"], lab["input_ids"])
    ]


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def collate(batch, pad_id, label_pad=-100):
    max_src = max(len(x["input_ids"]) for x in batch)
    max_tgt = max(len(x["labels"]) for x in batch)
    input_ids = torch.full((len(batch), max_src), pad_id, dtype=torch.long)
    attention = torch.zeros((len(batch), max_src), dtype=torch.long)
    labels = torch.full((len(batch), max_tgt), label_pad, dtype=torch.long)
    for i, row in enumerate(batch):
        n, m = len(row["input_ids"]), len(row["labels"])
        input_ids[i, :n] = torch.tensor(row["input_ids"])
        attention[i, :n] = torch.tensor(row["attention_mask"])
        labels[i, :m] = torch.tensor(row["labels"])
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


def batches(rows, batch_size, shuffle=False, seed=0):
    order = list(range(len(rows)))
    if shuffle:
        import random

        random.Random(seed).shuffle(order)
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


def evaluate_loss(model, rows, batch_size, pad_id, device):
    model.eval()
    total, tokens = 0.0, 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for idx in batches(rows, batch_size):
            batch = collate([rows[i] for i in idx], pad_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            n = int((batch["labels"] != -100).sum())
            total += float(out.loss) * n
            tokens += n
    model.train()
    return total / max(1, tokens)


@torch.no_grad()
def generate(model, tok, srcs, prefix, cfg, device, batch_size=16, num_beams=4):
    hyps: list[str] = []
    t0 = time.perf_counter()
    model.eval()
    for start in range(0, len(srcs), batch_size):
        chunk = [prefix + s for s in srcs[start : start + batch_size]]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=cfg["max_src_tokens"]).to(device)
        out = model.generate(**enc, max_new_tokens=cfg["predict_max_new_tokens"], num_beams=num_beams,
                             length_penalty=0.8, early_stopping=True)
        hyps.extend(tok.batch_decode(out, skip_special_tokens=True))
    return hyps, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--output-dir", default=str(OUT_DIR),
                    help="checkpoint/history output directory (relative paths are project-rooted)")
    ap.add_argument("--skip-zero-shot", action="store_true",
                    help="skip the pretrained zero-shot evaluation when it is already recorded")
    ap.add_argument("--train-only", action="store_true",
                    help="save training outputs and skip test-set inference; use the fixed-sample inference script later")
    ap.add_argument("--stop-after-epoch", type=int, default=None,
                    help="stop after this epoch while keeping the schedule sized for --epochs")
    args = ap.parse_args()
    cfg = load_config(args.config)
    mcfg = dict(cfg["mt5"])
    if args.epochs:
        mcfg["epochs"] = args.epochs
    set_seed(cfg["seed"])
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = project_path(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger("train_mt5", project_path("logs", "train_mt5.log"))
    jsonl = JsonlLogger(project_path("logs", "train_mt5.jsonl"))
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, project_path("logs", "config_used_mt5.yaml"))

    tok = AutoTokenizer.from_pretrained(mcfg["model_id"], revision=mcfg["revision"])
    model = AutoModelForSeq2SeqLM.from_pretrained(mcfg["model_id"], revision=mcfg["revision"])
    model.config.use_cache = False
    if mcfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()

    train_rows_raw = load_split("train")
    val_rows_raw = load_split("validation")
    test_rows = load_split("test")
    if args.smoke:
        train_rows_raw, val_rows_raw, test_rows = train_rows_raw[:200], val_rows_raw[:20], test_rows[:10]
        mcfg.update(epochs=1, batch_size=4, grad_accum=1)

    t0 = time.perf_counter()
    train_rows = tokenize_rows(train_rows_raw, tok, mcfg["source_prefix"], mcfg["max_src_tokens"], mcfg["max_tgt_tokens"])
    val_rows = tokenize_rows(val_rows_raw, tok, mcfg["source_prefix"], mcfg["max_src_tokens"], mcfg["max_tgt_tokens"])
    logger.info("tokenised %d train / %d val pairs in %.1fs", len(train_rows), len(val_rows), time.perf_counter() - t0)

    params_total = sum(p.numel() for p in model.parameters())
    logger.info("mT5 parameters: %s", f"{params_total:,}")
    model.to(device)

    # zero-shot ablation with the *pretrained* weights, before any fine-tuning
    if not args.smoke and not args.skip_zero_shot:
        zs_srcs = [r["en"] for r in test_rows[:300]]
        zs_refs = [r["hi"] for r in test_rows[:300]]
        zs_hyps, zs_seconds = generate(model, tok, zs_srcs, mcfg["source_prefix"], mcfg, device,
                                       batch_size=16, num_beams=1)
        zs_scores = corpus_scores(zs_hyps, zs_refs, cfg["evaluation"]["bleu_tokenizer"])
        write_json(RESULTS / "metrics_mt5_base_zeroshot300.json", {
            "system": "mt5_base", "variant": "zeroshot300_pretrained", "metrics": zs_scores,
            "n_test": len(zs_srcs), "inference_seconds": round(zs_seconds, 2),
            "note": "pretrained google/mt5-small, greedy decoding, same 300 test sentences"})
        logger.info("zero-shot mT5 BLEU %.2f chrF %.2f", zs_scores["bleu"], zs_scores["chrf"])

    opt_cls = torch.optim.AdamW
    if mcfg.get("optim") == "adafactor":
        from transformers.optimization import Adafactor

        optimizer = Adafactor([p for p in model.parameters() if p.requires_grad], lr=mcfg["lr"], scale_parameter=False,
                              relative_step=False, warmup_init=False, weight_decay=mcfg["weight_decay"], clip_threshold=1.0)
    else:
        optimizer = opt_cls(model.parameters(), lr=mcfg["lr"], weight_decay=mcfg["weight_decay"])

    steps_per_epoch = math.ceil(len(train_rows) / (mcfg["batch_size"] * mcfg["grad_accum"]))
    total_steps = max(1, steps_per_epoch * mcfg["epochs"])
    warmup = max(1, min(mcfg["warmup_steps"], total_steps // 10))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        return max(0.05, (warmup / step) ** 0.5)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    best_val = float("inf")
    history = []
    global_step = 0
    train_start = time.perf_counter()
    for epoch in range(1, mcfg["epochs"] + 1):
        model.train()
        running, seen = 0.0, 0
        t_epoch = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for step, idx in enumerate(batches(train_rows, mcfg["batch_size"], shuffle=True, seed=cfg["seed"] + epoch)):
            batch = collate([train_rows[i] for i in idx], pad_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(**batch)
                loss = out.loss / mcfg["grad_accum"]
            loss.backward()
            running += float(out.loss.detach()) * len(idx)
            seen += len(idx)
            if (step + 1) % mcfg["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 50 == 0:
                    jsonl.log(event="train_step", step=global_step, epoch=epoch, loss=round(float(out.loss.detach()), 4),
                              lr=round(optimizer.param_groups[0]["lr"], 8))
        val_loss = evaluate_loss(model, val_rows, mcfg["batch_size"], pad_id, device)
        row = {"epoch": epoch, "train_loss": round(running / max(1, seen), 4), "val_loss": round(val_loss, 4),
               "epoch_seconds": round(time.perf_counter() - t_epoch, 1)}
        history.append(row)
        logger.info("epoch %d | train %.4f | val %.4f | %.1fs", epoch, row["train_loss"], val_loss, row["epoch_seconds"])
        jsonl.log(event="epoch", **row)
        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(out_dir / "best")
            tok.save_pretrained(out_dir / "best")
            logger.info("saved best mT5 checkpoint (val loss %.4f)", val_loss)
        if args.stop_after_epoch and epoch >= args.stop_after_epoch:
            logger.info("stopping after epoch %d by request (schedule sized for %d epochs)", epoch, mcfg["epochs"])
            break

    train_seconds = time.perf_counter() - train_start
    write_json(out_dir / "training_history.json", {
        "history": history,
        "train_seconds": round(train_seconds, 1),
        "params_total": params_total,
        "total_steps": global_step,
        "planned_epochs": mcfg["epochs"],
        "starting_checkpoint": f"{mcfg['model_id']}@{mcfg['revision']}",
    })

    if args.train_only:
        logger.info("training-only run complete; test evaluation is left to the fixed-sample inference stage")
        return 0

    # best checkpoint for test inference
    model = AutoModelForSeq2SeqLM.from_pretrained(out_dir / "best")
    model.to(device)
    model.eval()

    srcs = [r["en"] for r in test_rows]
    refs = [r["hi"] for r in test_rows]
    hyps, infer_seconds = generate(model, tok, srcs, mcfg["source_prefix"], mcfg, device,
                                   batch_size=4, num_beams=mcfg["num_beams"])
    scores = corpus_scores(hyps, refs, cfg["evaluation"]["bleu_tokenizer"])
    diag = output_diagnostics(hyps, srcs)
    epochs_done = len(history)
    result = {
        "system": "mt5_base",
        "variant": "finetuned_epoch1" if epochs_done == 1 else "finetuned",
        "metrics": scores,
        "diagnostics": diag,
        "timing": {"train_seconds": round(train_seconds, 1),
                   "train_seconds_per_epoch": round(train_seconds / max(1, epochs_done), 1),
                   "inference_seconds": round(infer_seconds, 2),
                   "sentences_per_second": round(len(hyps) / max(infer_seconds, 1e-6), 2),
                   "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                   "batch_size": 4,
                   "decoding": f"beam={mcfg['num_beams']}"},
        "params_total": params_total,
        "checkpoint": str((out_dir / "best").relative_to(project_path())),
        "config": mcfg,
        "n_test": len(hyps),
        "epochs_completed": epochs_done,
        "planned_epochs": mcfg["epochs"],
        "early_stop_note": ("Training stopped after epoch 1 to keep the single-GPU schedule; "
                            "the epoch-1 best checkpoint is evaluated.") if epochs_done == 1 else None,
        "smoke": args.smoke,
    }
    write_json(RESULTS / "metrics_mt5_base.json", result)
    write_jsonl(TRANSLATIONS / "mt5_base.jsonl",
                [{"i": i, "src": s, "ref": r, "hyp": h} for i, (s, r, h) in enumerate(zip(srcs, refs, hyps))])
    logger.info("TEST %s", json.dumps(scores))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
