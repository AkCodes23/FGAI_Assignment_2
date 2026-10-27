#!/usr/bin/env python
"""Sampling-temperature study for decoder-only MT systems."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from fgai_mt.config import load_config, project_path  # noqa: E402
from fgai_mt.eval_sample import load_eval_rows  # noqa: E402
from fgai_mt.decoder import generate_texts, load_model, load_tokenizer, render_prompt  # noqa: E402
from fgai_mt.metrics import corpus_scores, output_diagnostics  # noqa: E402
from fgai_mt.utils import get_logger, read_json, set_seed, write_json, write_jsonl  # noqa: E402

RESULTS = project_path("results")
OUT = RESULTS / "temperature"


def distinct_n(texts, n=1):
    grams = Counter()
    total = 0
    for text in texts:
        toks = text.split()
        for i in range(len(toks) - n + 1):
            grams[tuple(toks[i : i + n])] += 1
            total += 1
    return round(len(grams) / max(1, total), 4)


def max_repeat(text: str) -> int:
    toks = text.split()
    best = 1
    for n in range(1, 6):
        for i in range(len(toks) - n + 1):
            gram = tuple(toks[i : i + n])
            run = 1
            j = i + n
            while j + n <= len(toks) and tuple(toks[j : j + n]) == gram:
                run += 1
                j += n
            best = max(best, run)
    return best


def load_system(key: str, cfg: dict):
    spec = cfg["decoder"][key]
    qcfg = cfg["decoder"]["qlora"]
    base = load_model(spec["model_id"], spec["revision"], load_in_4bit=qcfg["load_in_4bit"])
    base.eval()
    tok = load_tokenizer(spec["model_id"], spec["revision"])
    return base, tok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--systems", nargs="*", default=["llama_prompt3shot", "gemma_prompt3shot"])
    args = ap.parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])
    logger = get_logger("temperature", project_path("logs", "temperature.log"))
    OUT.mkdir(parents=True, exist_ok=True)
    indexed_test = load_eval_rows(args.limit or int(cfg["evaluation"]["temperature_sample_size"]))
    test_indices = [i for i, _ in indexed_test]
    test = [row for _, row in indexed_test]
    srcs = [r["en"] for r in test]
    refs = [r["hi"] for r in test]
    shots = read_json(RESULTS / "prompt_shots.json")["shots"]
    temps = cfg["evaluation"]["temperature_grid"]

    all_results = {}
    for tag in args.systems:
        key = tag.removesuffix("_prompt3shot")
        if key not in ("llama", "gemma"):
            raise ValueError(f"temperature analysis supports prompt-only systems, got {tag}")
        model, tok = load_system(key, cfg)
        prompts = [render_prompt(tok, s, shots=shots) for s in srcs]
        rows = []
        for temp in temps:
            set_seed(cfg["seed"])
            gen = generate_texts(model, tok, prompts, max_new_tokens=cfg["decoder"]["max_new_tokens"],
                                 batch_size=cfg["decoder"]["batch_size"], temperature=temp,
                                 top_p=cfg["evaluation"]["top_p"], max_input_tokens=cfg["decoder"]["max_input_tokens"])
            hyps = gen["texts"]
            scores = corpus_scores(hyps, refs, cfg["evaluation"]["bleu_tokenizer"])
            diag = output_diagnostics(hyps, srcs)
            entry = {
                "temperature": temp,
                "bleu": scores["bleu"],
                "chrf": scores["chrf"],
                "ter": scores["ter"],
                "distinct_1": distinct_n(hyps, 1),
                "distinct_2": distinct_n(hyps, 2),
                "mean_words": round(sum(len(h.split()) for h in hyps) / len(hyps), 2),
                "mean_max_token_repeat": round(sum(max_repeat(h) for h in hyps) / len(hyps), 3),
                "empty_rate": diag["empty_rate"],
                "no_devanagari_rate": diag["no_devanagari_rate"],
                "seconds": gen["seconds"],
                "tokens_per_second": gen["tokens_per_second"],
                "hit_max_new_tokens": gen["hit_max_new_tokens"],
            }
            rows.append(entry)
            logger.info("[%s] temp=%s BLEU=%.2f chrF=%.2f distinct2=%.3f", tag, entry["temperature"],
                        entry["bleu"], entry["chrf"], entry["distinct_2"])
            write_json(OUT / f"{tag}_temp{entry['temperature']}.json",
                       {"settings": entry, "n": len(hyps), "test_row_indices": test_indices})
            write_jsonl(OUT / f"{tag}_temp{entry['temperature']}.jsonl",
                        [{"i": i, "src": s, "ref": r, "hyp": h}
                         for i, s, r, h in zip(test_indices, srcs, refs, hyps)])
        all_results[tag] = rows
        del model
        torch.cuda.empty_cache()

    write_json(OUT / "temperature_results.json", {
        "n": len(test_indices), "test_row_indices": test_indices, "results": all_results,
        "prompt_shots": len(shots), "seed": cfg["seed"],
    })

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for tag, rows in all_results.items():
        x = [r["temperature"] for r in rows]
        label = tag.replace("_", " ")
        axes[0].plot(x, [r["bleu"] for r in rows], marker="o", label=label)
        axes[1].plot(x, [r["distinct_2"] for r in rows], marker="o", label=label)
    axes[0].set_xlabel("temperature")
    axes[0].set_ylabel("BLEU")
    axes[1].set_xlabel("temperature")
    axes[1].set_ylabel("distinct-2 (higher = more varied)")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Sampling temperature vs translation quality and diversity")
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "temperature_analysis.png")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
