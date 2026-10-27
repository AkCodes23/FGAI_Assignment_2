#!/usr/bin/env python
"""Tokenizer fertility on the held-out test references (no model inference).

Fertility is the number of subword tokens a tokenizer needs per Hindi sentence.
It explains part of the decoder-only inference-cost gap and the risk that a
fixed ``max_new_tokens`` cap truncates the more fragmented tokenizer.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import AutoTokenizer  # noqa: E402

from fgai_mt.config import load_config, project_path  # noqa: E402
from fgai_mt.data import load_split  # noqa: E402
from fgai_mt.utils import write_json  # noqa: E402


def main() -> int:
    cfg = load_config()
    rows = load_split("test")
    hi = [r["hi"] for r in rows]
    en = [r["en"] for r in rows]
    out = {}
    for key in ("llama", "gemma"):
        spec = cfg["decoder"][key]
        tok = AutoTokenizer.from_pretrained(spec["model_id"], revision=spec["revision"])
        hi_tokens = [len(tok(x, add_special_tokens=False)["input_ids"]) for x in hi]
        en_tokens = [len(tok(x, add_special_tokens=False)["input_ids"]) for x in en]
        out[spec["model_id"]] = {
            "n_sentences": len(rows),
            "hi_tokens_mean": round(statistics.fmean(hi_tokens), 2),
            "hi_tokens_median": statistics.median(hi_tokens),
            "hi_tokens_p95": sorted(hi_tokens)[int(0.95 * len(hi_tokens)) - 1],
            "en_tokens_mean": round(statistics.fmean(en_tokens), 2),
            "hi_words_mean": round(statistics.fmean(len(x.split()) for x in hi), 2),
            "hi_tokens_per_word": round(statistics.fmean(hi_tokens) / statistics.fmean(len(x.split()) for x in hi), 3),
            "max_new_tokens_cap": cfg["decoder"]["max_new_tokens"],
            "references_over_cap": sum(1 for n in hi_tokens if n > cfg["decoder"]["max_new_tokens"]),
        }
        print(spec["model_id"], out[spec["model_id"]])
    write_json(project_path("results", "tokenizer_fertility.json"), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
