#!/usr/bin/env python
"""Inference-only pass for the from-scratch Transformer (loads models/basic_transformer/best.pt)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from fgai_mt.config import load_config, project_path  # noqa: E402
from fgai_mt.eval_sample import load_eval_rows  # noqa: E402
from fgai_mt.metrics import corpus_scores, output_diagnostics  # noqa: E402
from fgai_mt.model_basic import TransformerMT, beam_search  # noqa: E402
from fgai_mt.tokenization import decode_ids, load_spm  # noqa: E402
from fgai_mt.utils import get_logger, read_json, write_json, write_jsonl  # noqa: E402

PAD, BOS, EOS = 0, 1, 2
OUT_DIR = project_path("models", "basic_transformer")


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger("infer_basic", project_path("logs", "infer_basic.log"))

    ckpt = torch.load(OUT_DIR / "best.pt", map_location=device, weights_only=False)
    bcfg = ckpt.get("config", cfg["basic_transformer"])
    model = TransformerMT(vocab_size=ckpt["vocab_size"], d_model=bcfg["d_model"], n_heads=bcfg["n_heads"],
                          n_encoder_layers=bcfg["n_encoder_layers"], n_decoder_layers=bcfg["n_decoder_layers"],
                          d_ff=bcfg["d_ff"], dropout=bcfg["dropout"], pad_id=PAD).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = load_spm(project_path("models", "spm_basic.model"))

    indexed_rows = load_eval_rows(args.limit)
    test_indices = [i for i, _ in indexed_rows]
    rows = [row for _, row in indexed_rows]
    srcs = [r["en"] for r in rows]
    refs = [r["hi"] for r in rows]
    hyps: list[str] = []
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for start in range(0, len(srcs), args.batch_size):
        chunk = srcs[start : start + args.batch_size]
        ids = [[BOS] + tok.encode(s, out_type=int)[: cfg["dataset"]["max_src_tokens"]] + [EOS] for s in chunk]
        width = max(len(x) for x in ids)
        src = torch.full((len(ids), width), PAD, dtype=torch.long)
        for j, x in enumerate(ids):
            src[j, : len(x)] = torch.tensor(x)
        out = beam_search(model.to(device), src.to(device), beam_size=bcfg["beam_size"],
                          max_len=cfg["dataset"]["max_tgt_tokens"] + 4, pad_id=PAD, bos_id=BOS, eos_id=EOS,
                          length_penalty=bcfg["length_penalty"], device=device)
        hyps.extend(decode_ids(tok, o) for o in out)
    seconds = time.perf_counter() - t0

    scores = corpus_scores(hyps, refs, cfg["evaluation"]["bleu_tokenizer"])
    diag = output_diagnostics(hyps, srcs)
    history = read_json(OUT_DIR / "training_history.json")
    epochs_completed = len(history.get("history", []))
    train_seconds = history.get("train_seconds")
    result = {
        "system": "basic_transformer",
        "variant": "finetuned",
        "metrics": scores,
        "diagnostics": diag,
        "timing": {
            "train_seconds": train_seconds,
            "train_seconds_per_epoch": round(train_seconds / max(1, epochs_completed), 1) if train_seconds else None,
            "inference_seconds": round(seconds, 2),
            "sentences_per_second": round(len(hyps) / max(seconds, 1e-9), 2),
            "new_tokens": None,
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else 0.0,
            "batch_size": args.batch_size,
            "decoding": f"beam={bcfg['beam_size']}, length_penalty={bcfg['length_penalty']}",
        },
        "params_total": sum(p.numel() for p in model.parameters()),
        "checkpoint": str((OUT_DIR / "best.pt").relative_to(project_path())),
        "config": bcfg,
        "n_test": len(hyps),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_loss": ckpt.get("val_loss"),
        "train_rows": ckpt.get("train_rows", history.get("train_rows")),
        "train_subset_sha256": ckpt.get("train_subset_sha256", history.get("train_subset_sha256")),
        "smoke": False,
    }
    write_json(project_path("results", "metrics_basic_transformer.json"), result)
    write_jsonl(project_path("results", "translations", "basic_transformer.jsonl"),
                [{"i": i, "src": s, "ref": r, "hyp": h}
                 for i, s, r, h in zip(test_indices, srcs, refs, hyps)])
    logger.info("TEST %s | %.1fs | %.2f sent/s | epoch %s", json.dumps(scores), seconds,
                len(hyps) / max(seconds, 1e-9), ckpt.get("epoch"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
