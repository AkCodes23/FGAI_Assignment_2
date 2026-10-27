#!/usr/bin/env python
"""Sanity check: can the from-scratch Transformer overfit a tiny training subset?

A healthy implementation drives training CE towards ~0 on a 2k-pair subset while validation CE
stays high. Run before trusting the full training run.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from fgai_mt.config import load_config, project_path
from fgai_mt.data import load_split
from fgai_mt.model_basic import TransformerMT
from fgai_mt.tokenization import load_spm

PAD, BOS, EOS = 0, 1, 2


def encode(rows, tok, max_src=96, max_tgt=96):
    return [([BOS] + tok.encode(r["en"], out_type=int)[:max_src] + [EOS],
             tok.encode(r["hi"], out_type=int)[:max_tgt] + [EOS]) for r in rows]


def collate(batch):
    ms = max(len(s) for s, _ in batch)
    mt = max(len(t) for _, t in batch)
    src = torch.full((len(batch), ms), PAD, dtype=torch.long)
    tgt_in = torch.full((len(batch), mt), PAD, dtype=torch.long)
    tgt_out = torch.full((len(batch), mt), PAD, dtype=torch.long)
    for i, (s, t) in enumerate(batch):
        src[i, : len(s)] = torch.tensor(s)
        tgt_in[i, : len(t)] = torch.tensor([BOS] + t[:-1])
        tgt_out[i, : len(t)] = torch.tensor(t)
    return src, tgt_in, tgt_out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    cfg = load_config()
    b = cfg["basic_transformer"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(0)
    torch.manual_seed(0)

    tok = load_spm()
    train = encode(load_split("train")[: args.pairs], tok)
    val = encode(load_split("validation")[: 200], tok)
    model = TransformerMT(vocab_size=tok.get_piece_size(), d_model=b["d_model"], n_heads=b["n_heads"],
                          n_encoder_layers=b["n_encoder_layers"], n_decoder_layers=b["n_decoder_layers"],
                          d_ff=b["d_ff"], dropout=b["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    steps = math.ceil(len(train) / args.batch) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200))
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)

    def epoch_loss(rows):
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(rows), 64):
                src, ti, to = (x.to(device) for x in collate(rows[i : i + 64]))
                out = model(src, ti)
                tot += float(loss_fn(out.reshape(-1, out.size(-1)), to.reshape(-1))) * (to != PAD).sum()
                n += int((to != PAD).sum())
        model.train()
        return tot / max(1, n)

    for epoch in range(1, args.epochs + 1):
        model.train()
        idx = list(range(len(train)))
        random.shuffle(idx)
        for i in range(0, len(idx), args.batch):
            src, ti, to = (x.to(device) for x in collate([train[j] for j in idx[i : i + args.batch]]))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out = model(src, ti)
                loss = loss_fn(out.reshape(-1, out.size(-1)), to.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} | train CE {epoch_loss(train[:500]):.3f} | val CE {epoch_loss(val):.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
