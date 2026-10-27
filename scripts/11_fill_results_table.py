#!/usr/bin/env python
"""Insert the generated headline table into README.md and docs/report.md.

The prose around the table is written by a human after inspecting the artefacts;
this script only prevents transcription errors in the table itself.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fgai_mt.config import project_path  # noqa: E402

ROOT = project_path()
RESULTS = ROOT / "results"


def main() -> int:
    table_path = RESULTS / "summary_table.md"
    if not table_path.exists():
        print("summary_table.md missing; run scripts/05_evaluate.py first")
        return 1
    table = table_path.read_text(encoding="utf-8").strip()
    facts_path = RESULTS / "analysis_facts.json"
    facts = json.loads(facts_path.read_text(encoding="utf-8")) if facts_path.exists() else {}
    sample_path = RESULTS / "evaluation_sample.json"
    sample = json.loads(sample_path.read_text(encoding="utf-8")) if sample_path.exists() else {}
    best = facts.get("best") or {}
    best_line = (f"**Best BLEU:** `{best.get('tag')}` at {best.get('bleu'):.2f} BLEU / "
                 f"{best.get('chrf'):.2f} chrF / {best.get('ter'):.1f} TER.") if best else ""

    block = (
        f"**Headline test-set results ({sample.get('sample_size', 'unknown')} fixed random rows; seed {sample.get('sample_seed', 'unknown')}).** "
        "BLEU is SacreBLEU 13a; the sample is drawn from the official IITB test split.\n"
        "chrF uses character order 6 and word order 0; TER is lower-is-better. The 95 % interval is a\n"
        "sentence-level bootstrap (400 resamples, fixed seed). `Train/adapt` is wall-clock seconds; `—` means the\n"
        "row did not train (prompt-only). `Batch` and `Decoding` are the inference settings actually used, because\n"
        "throughput is only comparable when those are stated.\n\n"
        f"{table}\n"
    )
    if best_line:
        block += f"\n{best_line}\n"

    changed = []
    for name in ("README.md", "docs/report.md"):
        path = ROOT / name
        text = path.read_text(encoding="utf-8")
        if ">>RESULTS_TABLE<<" not in text:
            print(f"{name}: no RESULTS_TABLE placeholder")
            continue
        path.write_text(text.replace(">>RESULTS_TABLE<<", block), encoding="utf-8")
        changed.append(name)
    print("inserted table into:", ", ".join(changed) if changed else "nothing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
