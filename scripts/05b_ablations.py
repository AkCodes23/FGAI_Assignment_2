#!/usr/bin/env python
"""Ablations that are not part of the headline table: demonstrations and mT5 fine-tuning."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fgai_mt.config import load_config, project_path  # noqa: E402
from fgai_mt.metrics import corpus_scores, paired_bootstrap_bleu  # noqa: E402
from fgai_mt.utils import read_json, read_jsonl, write_json  # noqa: E402

RESULTS = project_path("results")
TRANS = RESULTS / "translations"


def main() -> int:
    cfg = load_config()
    tok = cfg["evaluation"]["bleu_tokenizer"]
    out: dict[str, object] = {}

    for family in ("llama", "gemma"):
        full = read_jsonl(TRANS / f"{family}_prompt3shot.jsonl")
        zero = read_jsonl(TRANS / f"{family}_prompt0shot.jsonl")
        n = len(zero)
        assert n > 0, family
        assert [(r["src"], r["ref"]) for r in full[:n]] == [(r["src"], r["ref"]) for r in zero]
        s3 = corpus_scores([r["hyp"] for r in full[:n]], [r["ref"] for r in full[:n]], tok)
        s0 = corpus_scores([r["hyp"] for r in zero], [r["ref"] for r in zero], tok)
        delta = paired_bootstrap_bleu([r["hyp"] for r in full[:n]], [r["hyp"] for r in zero],
                                      [r["ref"] for r in zero], n_samples=400, seed=7)
        out[f"{family}_demonstrations"] = {
            "n": n,
            "prompt3shot": s3,
            "prompt0shot": s0,
            "paired_3shot_minus_0shot": delta,
        }

    mt5 = read_jsonl(TRANS / "mt5_base.jsonl")
    zero_path = RESULTS / "metrics_mt5_base_zeroshot300.json"
    if zero_path.exists():
        zero = read_json(zero_path)
        n = int(zero.get("n_test", 300))
        rows = mt5[:n]
        fine = corpus_scores([r["hyp"] for r in rows], [r["ref"] for r in rows], tok)
        out["mt5_finetune"] = {
            "n": n,
            "zeroshot_metrics": zero["metrics"],
            "finetuned_metrics_same_subset": fine,
            "note": "Zero-shot outputs were not saved, so this is a metric comparison on the same first "
                    "test sentences rather than a paired bootstrap.",
        }

    write_json(RESULTS / "ablations.json", out)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
