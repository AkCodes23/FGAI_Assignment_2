#!/usr/bin/env python
"""Build a fixed, human-readable sample sheet from saved test translations.

The sample indices are fixed in advance so the sheet is reproducible and cannot
be curated after seeing which system looks best. It is a reading aid for the
report, not a metric: every row comes from the same official test split used in
the headline table.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fgai_mt.config import project_path  # noqa: E402
from fgai_mt.utils import read_json  # noqa: E402
from fgai_mt.utils import read_jsonl  # noqa: E402

ROOT = project_path()
RESULTS = ROOT / "results"
TRANS = RESULTS / "translations"

SYSTEMS = [
    ("basic_transformer", "Basic Transformer (scratch)"),
    ("mt5_base", "mT5-small (fine-tuned)"),
    ("llama_prompt3shot", "TinyLlama 1.1B (3-shot)"),
    ("gemma_prompt3shot", "Gemma 3 1B (3-shot)"),
]
# Fixed positions within the seeded 200-row sample, chosen before reading outputs.
SAMPLE_POSITIONS = [0, 17, 30, 73, 100, 150, 175, 199]


def main() -> int:
    loaded = {}
    base = None
    for tag, label in SYSTEMS:
        path = TRANS / f"{tag}.jsonl"
        if not path.exists():
            print(f"missing {tag}; skipped")
            continue
        rows = read_jsonl(path)
        loaded[tag] = (label, {r["i"]: r for r in rows})
        if base is None:
            base = rows
    if base is None:
        print("no translations found")
        return 1

    sample = read_json(RESULTS / "evaluation_sample.json")
    indices = [sample["test_row_indices"][position] for position in SAMPLE_POSITIONS]
    base_by_i = {r["i"]: r for r in loaded["basic_transformer"][1].values()}
    lines = [
        "# Qualitative translation samples",
        "",
        "Original test-row IDs selected by fixed positions in the precommitted seeded sample: "
        + ", ".join(str(i) for i in indices)
        + ". All systems use the same 200 sampled rows from the official test split.",
        "",
    ]
    csv_rows = []
    for i in indices:
        ref_row = base_by_i[i]
        lines += [f"## Test sentence {i}", "", f"**English:** {ref_row['src']}", "",
                  f"**Reference Hindi:** {ref_row['ref']}", "",
                  "| System | Hypothesis |", "|---|---|"]
        for tag, (label, by_i) in loaded.items():
            hyp = by_i.get(i, {}).get("hyp", "")
            hyp = hyp.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {label} | {hyp} |")
            csv_rows.append({"i": i, "src": ref_row["src"], "ref": ref_row["ref"],
                             "system": tag, "hyp": by_i.get(i, {}).get("hyp", "")})
        lines.append("")

    md_path = RESULTS / "qualitative_samples.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    with (RESULTS / "qualitative_samples.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["i", "src", "ref", "system", "hyp"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"wrote {md_path} and results/qualitative_samples.csv ({len(csv_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
