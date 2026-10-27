#!/usr/bin/env python
"""Aggregate metrics, confidence intervals, paired significance tests, tables and plots."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from fgai_mt.config import load_config, project_path  # noqa: E402
from fgai_mt.metrics import bootstrap_ci, paired_bootstrap_bleu  # noqa: E402
from fgai_mt.utils import read_json, read_jsonl, write_json  # noqa: E402

RESULTS = project_path("results")
FIGS = RESULTS / "figures"

SYSTEM_ORDER = [
    ("basic_transformer", "Basic Transformer (from scratch)"),
    ("mt5_base", "mT5-small (fine-tuned)"),
    ("llama_prompt3shot", "TinyLlama 1.1B (3-shot prompt)"),
    ("gemma_prompt3shot", "Gemma 3 1B (3-shot prompt)"),
]


def load_system(tag: str):
    trans = RESULTS / "translations" / f"{tag}.jsonl"
    if not trans.exists():
        return None
    rows = read_jsonl(trans)
    metrics_path = RESULTS / f"metrics_{tag}.json"
    metrics = read_json(metrics_path) if metrics_path.exists() else {}
    return rows, metrics


def paired_bootstrap(rows_a, rows_b, refs, n=400, seed=7):
    a = [r["hyp"] for r in rows_a]
    b = [r["hyp"] for r in rows_b]
    return paired_bootstrap_bleu(a, b, refs, n_samples=n, seed=seed)


def main() -> int:
    cfg = load_config()
    FIGS.mkdir(parents=True, exist_ok=True)
    table = []
    cis = {}
    systems = {}
    for tag, label in SYSTEM_ORDER:
        loaded = load_system(tag)
        if not loaded:
            print(f"missing {tag}")
            continue
        rows, metrics = loaded
        systems[tag] = rows
        hyps = [r["hyp"] for r in rows]
        refs = [r["ref"] for r in rows]
        srcs = [r["src"] for r in rows]
        if not metrics:
            continue
        if metrics.get("n_test") not in (None, len(rows)):
            raise SystemExit(f"{tag}: translations have {len(rows)} rows but metrics say {metrics.get('n_test')}")
        m = metrics["metrics"]
        timing = metrics.get("timing", {})
        print(f"bootstrapping {tag} ...", flush=True)
        bleu_ci = bootstrap_ci(hyps, refs, "bleu", cfg["evaluation"]["bootstrap_samples"])
        cis[tag] = {"bleu": bleu_ci}
        table.append({
            "tag": tag,
            "system": label,
            "architecture": "encoder-decoder" if tag in ("basic_transformer", "mt5_base") else "decoder-only",
            "params_m": round(metrics.get("params_total", 0) / 1e6, 1),
            "train_or_adapt_s": timing.get("train_seconds") or timing.get("adaptation_seconds"),
            "inference_s": timing.get("inference_seconds"),
            "sent_per_s": timing.get("sentences_per_second"),
            "inference_batch": timing.get("batch_size"),
            "decoding": timing.get("decoding"),
            "bleu": m["bleu"],
            "bleu_ci_lo": bleu_ci["lo"],
            "bleu_ci_hi": bleu_ci["hi"],
            "chrf": m["chrf"],
            "ter": m["ter"],
            "peak_vram_gb": timing.get("peak_vram_gb"),
            "n_test": metrics.get("n_test"),
            "empty_rate": metrics.get("diagnostics", {}).get("empty_rate"),
            "no_devanagari_rate": metrics.get("diagnostics", {}).get("no_devanagari_rate"),
            "length_ratio_mean": metrics.get("diagnostics", {}).get("length_ratio_mean"),
        })

    df = pd.DataFrame(table)
    df.to_csv(RESULTS / "summary_table.csv", index=False)
    write_json(RESULTS / "confidence_intervals.json", cis)

    # paired significance for the comparisons the report actually discusses
    pairs = []
    key_pairs = [
        ("basic_transformer", "mt5_base"),
        ("basic_transformer", "llama_prompt3shot"),
        ("basic_transformer", "gemma_prompt3shot"),
        ("mt5_base", "llama_prompt3shot"),
        ("mt5_base", "gemma_prompt3shot"),
        ("llama_prompt3shot", "gemma_prompt3shot"),
    ]
    for a, b in key_pairs:
        if a not in systems or b not in systems:
            continue
        if len(systems[a]) != len(systems[b]):
            continue
        refs = [r["ref"] for r in systems[a]]
        pairs.append({"a": a, "b": b, **paired_bootstrap(systems[a], systems[b], refs)})
    write_json(RESULTS / "paired_bootstrap.json", pairs)

    # markdown table
    lines = ["| System | Arch | Params (M) | Train/adapt (s) | Inference (s) | Sent/s | Batch | Decoding | BLEU | 95% CI | chrF | TER |",
             "|---|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|"]
    for row in table:
        lines.append(
            f"| {row['system']} | {row['architecture'].replace('encoder-decoder','enc-dec')} | {row['params_m']:.1f} | "
            f"{row['train_or_adapt_s'] if row['train_or_adapt_s'] is not None else '—'} | {row['inference_s']} | "
            f"{row['sent_per_s']:.2f} | {row['inference_batch'] if row['inference_batch'] is not None else '—'} | "
            f"{row['decoding'] or '—'} | "
            f"{row['bleu']:.2f} | [{row['bleu_ci_lo']:.2f}, {row['bleu_ci_hi']:.2f}] | {row['chrf']:.2f} | {row['ter']:.1f} |"
        )
    (RESULTS / "summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- figures ---------------------------------------------------------
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9})
    if table:
        short = [r["tag"].replace("_", " ") for r in table]
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        ax.bar(short, [r["bleu"] for r in table], color="#3b6ea5")
        ax.errorbar(short, [r["bleu"] for r in table],
                    yerr=[[r["bleu"] - r["bleu_ci_lo"] for r in table], [r["bleu_ci_hi"] - r["bleu"] for r in table]],
                    fmt="none", ecolor="black", capsize=3)
        ax.set_ylabel("BLEU (13a)")
        ax.set_title("English→Hindi test BLEU with 95% bootstrap CI")
        plt.xticks(rotation=25, ha="right")
        fig.tight_layout()
        fig.savefig(FIGS / "bleu_by_system.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        for row in table:
            ax.scatter(row["params_m"], row["bleu"], s=45)
            ax.annotate(row["tag"], (row["params_m"], row["bleu"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Parameters (millions, log scale)")
        ax.set_xscale("log")
        ax.set_ylabel("BLEU")
        ax.set_title("Quality vs model size")
        fig.tight_layout()
        fig.savefig(FIGS / "bleu_vs_params.png")
        plt.close(fig)

    print(df.to_string(index=False))
    print("\nWrote results/summary_table.md, summary_table.csv, figures/, confidence_intervals.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
