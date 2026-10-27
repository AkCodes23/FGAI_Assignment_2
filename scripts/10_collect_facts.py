#!/usr/bin/env python
"""Collect the numbers used in the README/report prose into one reviewable file.

Reads only saved artifacts; performs no model inference. The output is the
single place a human can check every figure quoted in the write-up.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fgai_mt.config import project_path  # noqa: E402


def read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def main() -> int:
    root = project_path()
    res = root / "results"
    table_path = res / "summary_table.csv"
    rows = list(csv.DictReader(table_path.open(encoding="utf-8"))) if table_path.exists() else []
    for row in rows:
        for key in ("params_m", "train_or_adapt_s", "inference_s", "sent_per_s", "bleu", "bleu_ci_lo",
                    "bleu_ci_hi", "chrf", "ter", "peak_vram_gb", "n_test", "empty_rate",
                    "no_devanagari_rate", "length_ratio_mean"):
            if row.get(key) not in (None, ""):
                try:
                    row[key] = float(row[key])
                except ValueError:
                    pass

    ranked = sorted([r for r in rows if isinstance(r.get("bleu"), float)], key=lambda r: r["bleu"], reverse=True)
    facts = {
        "n_systems": len(rows),
        "test_rows": rows[0].get("n_test") if rows else None,
        "ranking": [{"tag": r["tag"], "bleu": r["bleu"], "chrf": r["chrf"], "ter": r["ter"]} for r in ranked],
        "best": ranked[0] if ranked else None,
        "worst": ranked[-1] if ranked else None,
        "paired": read_json(res / "paired_bootstrap.json", []),
        "ablations": read_json(res / "ablations.json", {}),
        "temperature": read_json(res / "temperature" / "temperature_results.json", {}),
        "risk_flags": read_json(res / "error_analysis" / "risk_flags.json", {}),
        "tokenizer_fertility": read_json(res / "tokenizer_fertility.json", {}),
        "mt5_zeroshot": read_json(res / "metrics_mt5_base_zeroshot300.json", {}),
        "summary_rows": rows,
    }
    out = res / "analysis_facts.json"
    out.write_text(json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8")

    print("rank | system | BLEU | chrF | TER | params(M) | train/adapt(s) | sent/s")
    for i, r in enumerate(ranked, 1):
        print(f"{i:>4} | {r['tag']:<20} | {r['bleu']:6.2f} | {r['chrf']:6.2f} | {r['ter']:6.1f} | "
              f"{r.get('params_m'):>8} | {r.get('train_or_adapt_s'):>14} | {r.get('sent_per_s')}")

    print("\npaired BLEU differences (A - B):")
    for p in facts["paired"]:
        sig = "excludes 0" if (p["ci_lo"] > 0 or p["ci_hi"] < 0) else "includes 0"
        print(f"  {p['a']:<20} - {p['b']:<20} = {p['delta_bleu']:+.3f} "
              f"[{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}] p(d<=0)={p['p_delta_le_0']:.3f} ({sig})")

    print("\ntemperature:")
    for tag, records in facts["temperature"].get("results", {}).items():
        cells = ", ".join(f"{r['temperature']}: BLEU {r['bleu']:.2f} chrF {r['chrf']:.2f} d2 {r['distinct_2']:.3f}"
                          for r in records)
        print(f"  {tag}: {cells}")

    print("\nrisk flags:")
    for tag, flags in facts["risk_flags"].items():
        print(f"  {tag:<20} n={flags.get('n')} empty={flags.get('empty')} "
              f"no_dev={flags.get('no_devanagari')} dropped_num={flags.get('dropped_number_sentences')} "
              f"added_num={flags.get('added_number_sentences')} latin={flags.get('untranslated_latin_sentences')} "
              f"repeat={flags.get('repetition_loops')}")

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
