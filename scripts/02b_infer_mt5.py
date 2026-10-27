#!/usr/bin/env python
"""Test-only inference for an mT5 checkpoint saved by 02_train_mt5.py."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: E402

from fgai_mt.config import load_config, project_path  # noqa: E402
from fgai_mt.eval_sample import load_eval_rows  # noqa: E402
from fgai_mt.metrics import corpus_scores, output_diagnostics  # noqa: E402
from fgai_mt.utils import get_logger, read_jsonl, write_json, write_jsonl  # noqa: E402

OUT_DIR = project_path("models", "mt5_base")
RESULTS = project_path("results")
TRANSLATIONS = RESULTS / "translations"


@torch.no_grad()
def generate(model, tok, srcs, prefix, cfg, device, batch_size=8, num_beams=4):
    hyps: list[str] = []
    new_tokens = 0
    t0 = time.perf_counter()
    model.eval()
    for start in range(0, len(srcs), batch_size):
        chunk = [prefix + s for s in srcs[start : start + batch_size]]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=cfg["max_src_tokens"]).to(device)
        out = model.generate(**enc, max_new_tokens=cfg["predict_max_new_tokens"], num_beams=num_beams,
                             length_penalty=0.8, early_stopping=True)
        hyps.extend(tok.batch_decode(out, skip_special_tokens=True))
        new_tokens += int((out != tok.pad_token_id).sum())
    return hyps, time.perf_counter() - t0, new_tokens


def training_seconds() -> tuple[float | None, int]:
    """Use the history file when present, otherwise sum epoch events from the JSONL log."""
    hist = OUT_DIR / "training_history.json"
    if hist.exists():
        data = json.loads(hist.read_text(encoding="utf-8"))
        return data.get("train_seconds"), len(data.get("history", []))
    log = project_path("logs", "train_mt5.jsonl")
    total, epochs = 0.0, 0
    if log.exists():
        for row in read_jsonl(log):
            if row.get("event") == "epoch":
                total += float(row.get("epoch_seconds", row.get("seconds", 0.0)))
                epochs += 1
    return (round(total, 1) if epochs else None), epochs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(OUT_DIR / "best"))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="recompute even when a fresh fixed-sample metrics file already exists")
    args = ap.parse_args()
    cfg = load_config()
    mcfg = dict(cfg["mt5"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger("infer_mt5", project_path("logs", "infer_mt5.log"))

    ckpt = Path(args.checkpoint)
    metrics_path = RESULTS / "metrics_mt5_base.json"
    if not args.force and not args.limit and metrics_path.exists():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        ckpt_mtime = max(p.stat().st_mtime for p in ckpt.rglob("*") if p.is_file())
        if existing.get("n_test") == len(load_eval_rows()) and metrics_path.stat().st_mtime >= ckpt_mtime:
            # Reuse a fresh fixed-sample pass. Fill any metadata that
            # only this recovery pass knows, without spending another generation run.
            train_s, epochs_done = training_seconds()
            timing = existing.setdefault("timing", {})
            timing.setdefault("batch_size", args.batch_size)
            timing.setdefault("decoding", f"beam={mcfg['num_beams']}")
            if train_s and epochs_done:
                timing["train_seconds"] = train_s
                timing["train_seconds_per_epoch"] = round(train_s / epochs_done, 1)
                existing["epochs_completed"] = epochs_done
                existing["planned_epochs"] = mcfg["epochs"]
                existing["variant"] = "finetuned_epoch1" if epochs_done == 1 else "finetuned"
                existing["early_stop_note"] = (
                    "Training stopped after epoch 1 to keep the single-GPU schedule; "
                    "the epoch-1 best checkpoint is evaluated.") if epochs_done == 1 else None
                write_json(metrics_path, existing)
            logger.info("reused fresh fixed-sample pass from %s (metadata repaired; use --force to recompute)",
                        metrics_path)
            print(f"reusing fresh fixed-sample pass: {metrics_path}")
            return 0
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt).to(device)
    model.eval()
    params_total = sum(p.numel() for p in model.parameters())

    indexed_rows = load_eval_rows(args.limit)
    indices = [i for i, _ in indexed_rows]
    rows = [row for _, row in indexed_rows]
    srcs = [r["en"] for r in rows]
    refs = [r["hi"] for r in rows]
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    hyps, seconds, new_tokens = generate(model, tok, srcs, mcfg["source_prefix"], mcfg, device,
                                         batch_size=args.batch_size, num_beams=mcfg["num_beams"])
    scores = corpus_scores(hyps, refs, cfg["evaluation"]["bleu_tokenizer"])
    diag = output_diagnostics(hyps, srcs)
    train_s, epochs_done = training_seconds()
    result = {
        "system": "mt5_base",
        "variant": "finetuned_epoch1" if epochs_done == 1 else "finetuned",
        "metrics": scores,
        "diagnostics": diag,
        "timing": {
            "train_seconds": train_s,
            "train_seconds_per_epoch": round(train_s / epochs_done, 1) if train_s and epochs_done else None,
            "inference_seconds": round(seconds, 2),
            "sentences_per_second": round(len(hyps) / max(seconds, 1e-9), 2),
            "new_tokens": new_tokens,
            "tokens_per_second": round(new_tokens / max(seconds, 1e-9), 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else 0.0,
            "batch_size": args.batch_size,
            "decoding": f"beam={mcfg['num_beams']}",
        },
        "params_total": params_total,
        "checkpoint": str(ckpt.relative_to(project_path())),
        "config": mcfg,
        "n_test": len(hyps),
        "epochs_completed": epochs_done,
        "planned_epochs": cfg["mt5"]["epochs"],
        "early_stop_note": ("Training stopped after epoch 1 to keep the single-GPU schedule; "
                            "the epoch-1 best checkpoint is evaluated.") if epochs_done == 1 else None,
        "smoke": False,
    }
    write_json(RESULTS / "metrics_mt5_base.json", result)
    write_jsonl(TRANSLATIONS / "mt5_base.jsonl",
                [{"i": i, "src": s, "ref": r, "hyp": h} for i, s, r, h in zip(indices, srcs, refs, hyps)])
    logger.info("TEST %s | %.1fs | %.2f sent/s | epochs=%s", json.dumps(scores), seconds,
                len(hyps) / max(seconds, 1e-9), epochs_done)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
