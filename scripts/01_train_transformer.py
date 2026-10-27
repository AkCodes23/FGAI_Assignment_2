#!/usr/bin/env python
"""Train the from-scratch encoder-decoder Transformer (English -> Hindi)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from fgai_mt.config import dump_config, load_config, project_path  # noqa: E402
from fgai_mt.data import load_split  # noqa: E402
from fgai_mt.metrics import corpus_scores, output_diagnostics  # noqa: E402
from fgai_mt.model_basic import TransformerMT, beam_search, count_parameters, greedy_decode  # noqa: E402
from fgai_mt.tokenization import decode_ids, load_spm  # noqa: E402
from fgai_mt.utils import JsonlLogger, Timer, get_logger, read_jsonl, set_seed, write_json, write_jsonl  # noqa: E402

PAD, BOS, EOS, UNK = 0, 1, 2, 3


def make_batches(pairs, batch_size, shuffle_rng=None, bucket_window=2000):
    order = list(range(len(pairs)))
    if shuffle_rng is not None:
        shuffle_rng.shuffle(order)
        # local length bucketing: sort chunks of `bucket_window` by source length
        chunks = [order[i : i + bucket_window] for i in range(0, len(order), bucket_window)]
        order = [idx for chunk in chunks for idx in sorted(chunk, key=lambda i: len(pairs[i][0]))]
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


def collate(batch, pad_id=PAD):
    max_src = max(len(s) for s, _ in batch)
    max_tgt = max(len(t) for _, t in batch)
    src = torch.full((len(batch), max_src), pad_id, dtype=torch.long)
    tgt_in = torch.full((len(batch), max_tgt), pad_id, dtype=torch.long)
    tgt_out = torch.full((len(batch), max_tgt), pad_id, dtype=torch.long)
    for i, (s, t) in enumerate(batch):
        src[i, : len(s)] = torch.tensor(s)
        tgt_in[i, : len(t)] = torch.tensor([BOS] + t[:-1])
        tgt_out[i, : len(t)] = torch.tensor(t)
    return src, tgt_in, tgt_out


def evaluate_loss(model, pairs, batch_size, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1, reduction="sum")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for idx in make_batches(pairs, batch_size):
            batch = [pairs[i] for i in idx]
            src, tgt_in, tgt_out = (t.to(device) for t in collate(batch))
            logits = model(src, tgt_in)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            total_loss += float(loss)
            total_tokens += int((tgt_out != PAD).sum())
    model.train()
    return total_loss / max(1, total_tokens)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate the pipeline")
    ap.add_argument("--no-test", action="store_true", help="skip the built-in test pass (use 01b instead)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--train-limit", type=int, default=None,
                    help="deterministically sample this many rows from the existing shared train split")
    ap.add_argument("--patience", type=int, default=None,
                    help="stop after this many consecutive validation-loss misses")
    args = ap.parse_args()
    cfg = load_config(args.config)
    bcfg = dict(cfg["basic_transformer"])
    seed = cfg["seed"]
    if args.epochs is not None:
        bcfg["epochs"] = args.epochs
    if args.train_limit is not None:
        bcfg["train_limit"] = args.train_limit
    if args.patience is not None:
        bcfg["early_stopping_patience"] = args.patience
    cfg["basic_transformer"] = bcfg
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = project_path("logs")
    run_tag = "train_basic_reduced" if args.train_limit is not None else "train_basic"
    logger = get_logger(run_tag, log_dir / f"{run_tag}.log")
    jsonl = JsonlLogger(log_dir / f"{run_tag}.jsonl")
    out_dir = project_path("models", "basic_transformer")
    out_dir.mkdir(parents=True, exist_ok=True)
    config_log = "config_used_basic_reduced.yaml" if args.train_limit is not None else "config_used_basic.yaml"
    dump_config(cfg, log_dir / config_log)

    train_rows = load_split("train")
    train_row_indices = list(range(len(train_rows)))
    if args.train_limit is not None and args.train_limit < len(train_rows):
        import random
        train_row_indices = sorted(random.Random(seed).sample(range(len(train_rows)), args.train_limit))
        train_rows = [train_rows[i] for i in train_row_indices]
    train_subset_sha256 = hashlib.sha256(
        json.dumps(train_rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    val_rows = load_split("validation")
    if args.smoke:
        train_rows, val_rows = train_rows[:400], val_rows[:40]
        bcfg.update(epochs=1, batch_size=16, warmup_steps=10)

    tok = load_spm(project_path("models", "spm_basic.model"))
    def encode(rows):
        return [([BOS] + tok.encode(r["en"], out_type=int)[: cfg["dataset"]["max_src_tokens"]] + [EOS],
                 tok.encode(r["hi"], out_type=int)[: cfg["dataset"]["max_tgt_tokens"]] + [EOS]) for r in rows]

    t0 = time.perf_counter()
    train_pairs = encode(train_rows)
    val_pairs = encode(val_rows)
    logger.info("tokenised %d train / %d val pairs in %.1fs", len(train_pairs), len(val_pairs), time.perf_counter() - t0)

    model = TransformerMT(
        vocab_size=tok.get_piece_size(),
        d_model=bcfg["d_model"],
        n_heads=bcfg["n_heads"],
        n_encoder_layers=bcfg["n_encoder_layers"],
        n_decoder_layers=bcfg["n_decoder_layers"],
        d_ff=bcfg["d_ff"],
        dropout=bcfg["dropout"],
        pad_id=PAD,
    ).to(device)
    n_params = count_parameters(model)
    logger.info("model parameters: %s", n_params)
    write_json(out_dir / "model_summary.json", {"params_total": n_params, "config": bcfg, "vocab_size": tok.get_piece_size(),
                                                 "train_rows": len(train_rows),
                                                 "train_subset_sha256": train_subset_sha256})
    write_json(project_path("results", "basic_train_subset.json"), {
        "selection": "seeded uniform sample without replacement from the existing shared train.tsv",
        "seed": seed,
        "source_train_tsv_sha256": hashlib.sha256(project_path("data", "processed", "train.tsv").read_bytes()).hexdigest(),
        "sample_size": len(train_rows),
        "train_row_indices": train_row_indices,
        "train_subset_sha256": train_subset_sha256,
    })

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or name.endswith(".bias") else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
        lr=bcfg["lr"], betas=(0.9, 0.98), eps=1e-9,
    )

    steps_per_epoch = math.ceil(len(train_pairs) / bcfg["batch_size"])
    total_steps = max(1, steps_per_epoch * bcfg["epochs"])
    warmup = max(1, min(bcfg["warmup_steps"], total_steps // 10))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        return max(0.05, math.sqrt(warmup / step))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=bcfg["label_smoothing"])

    rng = __import__("random").Random(seed)
    best_val = float("inf")
    global_step = 0
    history = []
    epochs_without_improvement = 0
    train_start = time.perf_counter()

    for epoch in range(1, bcfg["epochs"] + 1):
        model.train()
        epoch_loss, seen = 0.0, 0
        t_epoch = time.perf_counter()
        for idx in make_batches(train_pairs, bcfg["batch_size"], shuffle_rng=rng):
            src, tgt_in, tgt_out = (t.to(device) for t in collate([train_pairs[i] for i in idx]))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(src, tgt_in)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), bcfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_loss += loss.item() * len(idx)
            seen += len(idx)
            if global_step % 100 == 0:
                jsonl.log(event="train_step", epoch=epoch, step=global_step, loss=round(loss.item(), 4),
                          lr=round(optimizer.param_groups[0]["lr"], 6))

        val_loss = evaluate_loss(model, val_pairs, bcfg["batch_size"], device)
        train_loss = epoch_loss / max(1, seen)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4),
                        "epoch_seconds": round(time.perf_counter() - t_epoch, 1)})
        logger.info("epoch %d | train %.4f | val %.4f | %.1fs", epoch, train_loss, val_loss, time.perf_counter() - t_epoch)
        jsonl.log(event="epoch", epoch=epoch, train_loss=round(train_loss, 4), val_loss=round(val_loss, 4),
                  seconds=round(time.perf_counter() - t_epoch, 1))
        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improvement = 0
            torch.save({"model": model.state_dict(), "config": bcfg, "vocab_size": tok.get_piece_size(), "epoch": epoch,
                        "val_loss": val_loss, "train_rows": len(train_rows),
                        "train_subset_sha256": train_subset_sha256}, out_dir / "best.pt")
            logger.info("saved new best checkpoint (val_loss %.4f)", val_loss)
        else:
            epochs_without_improvement += 1
            if args.patience is not None and epochs_without_improvement >= args.patience:
                logger.info("stopping after %d epochs without validation improvement", epochs_without_improvement)
                break

    train_seconds = time.perf_counter() - train_start
    write_json(out_dir / "training_history.json", {"history": history, "train_seconds": round(train_seconds, 1),
                                                   "params_total": n_params, "total_steps": global_step,
                                                   "train_rows": len(train_rows),
                                                   "train_subset_sha256": train_subset_sha256,
                                                   "epochs_completed": len(history),
                                                   "planned_epochs": bcfg["epochs"]})

    if args.no_test:
        logger.info("training complete; skipping built-in test pass (--no-test)")
        return 0

    # ---- final test inference -------------------------------------------
    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    test_rows = load_split("test")
    if args.smoke:
        test_rows = test_rows[:20]
    srcs = [r["en"] for r in test_rows]
    refs = [r["hi"] for r in test_rows]
    hyps, infer_seconds = translate_all(model, srcs, tok, device, {**bcfg,
                                                                     "max_src_tokens": cfg["dataset"]["max_src_tokens"],
                                                                     "max_tgt_tokens": cfg["dataset"]["max_tgt_tokens"]},
                                        batch_size=32)
    scores = corpus_scores(hyps, refs, cfg["evaluation"]["bleu_tokenizer"])
    diag = output_diagnostics(hyps, srcs)
    result = {
        "system": "basic_transformer",
        "metrics": scores,
        "diagnostics": diag,
        "timing": {"train_seconds": round(train_seconds, 1), "train_seconds_per_epoch": round(train_seconds / bcfg["epochs"], 1),
                   "inference_seconds": round(infer_seconds, 2),
                   "sentences_per_second": round(len(hyps) / max(infer_seconds, 1e-6), 2),
                   "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else 0.0},
        "params_total": n_params,
        "checkpoint": str((out_dir / "best.pt").relative_to(project_path())),
        "config": bcfg,
        "n_test": len(hyps),
        "smoke": args.smoke,
    }
    write_json(project_path("results", "metrics_basic_transformer.json"), result)
    write_jsonl(project_path("results", "translations", "basic_transformer.jsonl"),
                [{"i": i, "src": s, "ref": r, "hyp": h} for i, (s, r, h) in enumerate(zip(srcs, refs, hyps))])
    logger.info("TEST %s", json.dumps(scores))
    return 0


def translate_all(model, srcs, tok, device, cfg, batch_size=32):
    hyps: list[str] = []
    t0 = time.perf_counter()
    for i in range(0, len(srcs), batch_size):
        chunk = srcs[i : i + batch_size]
        ids = [[BOS] + tok.encode(s, out_type=int)[: cfg["max_src_tokens"]] + [EOS] for s in chunk]
        max_len = max(len(x) for x in ids)
        src = torch.full((len(ids), max_len), PAD, dtype=torch.long)
        for j, x in enumerate(ids):
            src[j, : len(x)] = torch.tensor(x)
        src = src.to(device)
        out = beam_search(model, src, beam_size=cfg["beam_size"], max_len=cfg["max_tgt_tokens"] + 4,
                          pad_id=PAD, bos_id=BOS, eos_id=EOS, length_penalty=cfg["length_penalty"], device=device)
        hyps.extend(decode_ids(tok, o) for o in out)
    return hyps, time.perf_counter() - t0


if __name__ == "__main__":
    raise SystemExit(main())

