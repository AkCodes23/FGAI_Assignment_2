#!/usr/bin/env python
"""Final reduced-scope check: required files, fixed sample, and cross-system alignment."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fgai_mt.config import project_path  # noqa: E402
from fgai_mt.utils import read_jsonl, write_json  # noqa: E402

ROOT = project_path()
RESULTS = ROOT / "results"
SYSTEMS = ["basic_transformer", "mt5_base", "llama_prompt3shot", "gemma_prompt3shot"]


def main() -> int:
    checks: dict[str, object] = {}
    required = [
        ROOT / "configs" / "default.yaml",
        ROOT / "data" / "processed" / "train.tsv",
        ROOT / "data" / "processed" / "validation.tsv",
        ROOT / "data" / "processed" / "test.tsv",
        ROOT / "data" / "processed" / "dataset_manifest.json",
        RESULTS / "evaluation_sample.json",
        ROOT / "models" / "spm_basic.model",
        ROOT / "models" / "basic_transformer" / "best.pt",
        ROOT / "models" / "mt5_base" / "best" / "config.json",
        RESULTS / "metrics_basic_transformer.json",
        RESULTS / "metrics_mt5_base.json",
        RESULTS / "metrics_llama_prompt3shot.json",
        RESULTS / "metrics_gemma_prompt3shot.json",
        RESULTS / "summary_table.md",
        RESULTS / "summary_table.csv",
        RESULTS / "README.md",
        RESULTS / "confidence_intervals.json",
        RESULTS / "paired_bootstrap.json",
        RESULTS / "qualitative_samples.md",
        RESULTS / "model_provenance.json",
        RESULTS / "error_analysis" / "risk_flags.json",
        RESULTS / "temperature" / "temperature_results.json",
        RESULTS / "figures" / "temperature_analysis.png",
        ROOT / "docs" / "report.md",
        ROOT / "docs" / "YourName_RegNum_FGAI_Assignment3_TEMPLATE.docx",
    ]
    checks["missing_files"] = [str(p.relative_to(ROOT)) for p in required if not p.exists()]

    sample_path = RESULTS / "evaluation_sample.json"
    sample = json.loads(sample_path.read_text(encoding="utf-8")) if sample_path.exists() else {}
    expected_ids = sample.get("test_row_indices", [])
    checks["sample_size"] = len(expected_ids)
    checks["sample_seed"] = sample.get("sample_seed")
    checks["sample_ids_unique"] = len(expected_ids) == len(set(expected_ids))

    alignment = {}
    ref_src = None
    for tag in SYSTEMS:
        path = RESULTS / "translations" / f"{tag}.jsonl"
        metric_path = RESULTS / f"metrics_{tag}.json"
        rows = read_jsonl(path) if path.exists() else []
        metrics = json.loads(metric_path.read_text(encoding="utf-8")) if metric_path.exists() else {}
        ids = [r.get("i") for r in rows]
        pairs = [(r.get("src"), r.get("ref")) for r in rows]
        if ref_src is None and rows:
            ref_src = pairs
        alignment[tag] = {
            "rows": len(rows),
            "metrics_n": metrics.get("n_test"),
            "ids_match_sample": ids == expected_ids,
            "sources_references_match": pairs == ref_src,
            "aligned": (len(rows) == len(expected_ids) and metrics.get("n_test") == len(expected_ids)
                        and ids == expected_ids and pairs == ref_src),
        }
    checks["translations"] = alignment

    manifest = ROOT / "data" / "processed" / "dataset_manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        checks["split_sizes"] = {k: v.get("pairs") for k, v in data.get("splits", {}).items()}
        checks["leakage_audit"] = data.get("leakage_audit")

    temp_path = RESULTS / "temperature" / "temperature_results.json"
    temp = json.loads(temp_path.read_text(encoding="utf-8")) if temp_path.exists() else {}
    temp_results = temp.get("results", {})
    expected_temp_ids = expected_ids[:20]
    checks["temperature"] = {
        "n": temp.get("n"),
        "ids_match_sample_prefix": temp.get("test_row_indices") == expected_temp_ids,
        "systems": sorted(temp_results),
        "three_temperatures_each": all(len(v) == 3 for v in temp_results.values()),
    }
    checks["pass"] = (not checks["missing_files"] and len(expected_ids) == 200
                      and checks["sample_ids_unique"]
                      and all(v.get("aligned") for v in alignment.values())
                      and checks["temperature"]["n"] == 20
                      and checks["temperature"]["ids_match_sample_prefix"]
                      and checks["temperature"]["three_temperatures_each"]
                      and set(temp_results) == {"llama_prompt3shot", "gemma_prompt3shot"})
    write_json(RESULTS / "artifact_check.json", checks)
    print(json.dumps(checks, indent=2, ensure_ascii=False)[:5000])
    return 0 if checks["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
