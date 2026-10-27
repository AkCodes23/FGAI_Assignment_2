#!/usr/bin/env python
"""Correct full parameter counts for quantized decoder-only runs without inference."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".hf"))
sys.path.insert(0, str(ROOT / "src"))
from fgai_mt.decoder import count_full_checkpoint_parameters  # noqa: E402

METHOD = "sum of original safetensors tensor shapes before 4-bit quantization"
TARGETS = {
    "llama_prompt3shot": "metrics_llama_prompt3shot.json",
    "gemma_prompt3shot": "metrics_gemma_prompt3shot.json",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    results = ROOT / "results"
    provenance_path = results / "model_provenance.json"
    provenance = read_json(provenance_path)
    provenance_by_role = {item["role"]: item for item in provenance["models"]}
    full_counts = {}
    for tag, filename in TARGETS.items():
        metric_path = results / filename
        metric = read_json(metric_path)
        model_id = metric["config"]["model_id"]
        revision = metric["config"].get("revision", "main")
        count = count_full_checkpoint_parameters(model_id, revision)
        full_counts[tag] = count
        metric["params_total"] = count
        metric["parameter_count_method"] = METHOD
        metric["parameter_count_resolved_commit"] = provenance_by_role[tag]["resolved_commit"]
        write_json(metric_path, metric)
        provenance_by_role[tag]["params_total"] = count
        provenance_by_role[tag]["parameter_count_method"] = METHOD
        provenance_by_role[tag]["parameter_count_source"] = "cached model.safetensors header at resolved_commit"
        print(f"{tag}: {count:,} parameters")
    write_json(provenance_path, provenance)

    csv_path = results / "summary_table.csv"
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if row["tag"] in full_counts:
            row["params_m"] = f"{full_counts[row['tag']] / 1e6:.1f}"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    md_path = results / "summary_table.md"
    md_lines = md_path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(md_lines):
        if line.startswith("| TinyLlama 1.1B (3-shot prompt) |"):
            cells = line.split("|")
            cells[3] = f" {full_counts['llama_prompt3shot'] / 1e6:.1f} "
            md_lines[index] = "|".join(cells)
        elif line.startswith("| Gemma 3 1B (3-shot prompt) |"):
            cells = line.split("|")
            cells[3] = f" {full_counts['gemma_prompt3shot'] / 1e6:.1f} "
            md_lines[index] = "|".join(cells)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    facts_path = results / "analysis_facts.json"
    facts = read_json(facts_path)
    for key in ("summary_rows",):
        for row in facts.get(key, []):
            if row.get("tag") in full_counts:
                row["params_m"] = round(full_counts[row["tag"]] / 1e6, 1)
    for key in ("best", "worst"):
        row = facts.get(key)
        if row and row.get("tag") in full_counts:
            row["params_m"] = round(full_counts[row["tag"]] / 1e6, 1)
    write_json(facts_path, facts)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for row in rows:
        params_m = float(row["params_m"])
        bleu = float(row["bleu"])
        ax.scatter(params_m, bleu, s=45)
        ax.annotate(row["tag"], (params_m, bleu), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Full model parameters (millions, log scale)")
    ax.set_xscale("log")
    ax.set_ylabel("BLEU")
    ax.set_title("Quality vs model size")
    fig.tight_layout()
    fig.savefig(results / "figures" / "bleu_vs_params.png", dpi=160)
    plt.close(fig)
    print("updated metrics, summary table, provenance, facts, and size plot; no model inference performed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
