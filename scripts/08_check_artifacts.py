#!/usr/bin/env python
"""Final reduced-scope check: fixed sample, EDA artifacts, and cross-system alignment."""
from __future__ import annotations

import json
import math
import re
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
    # Require only artifacts shipped in the sanitized public package. Training
    # data, model weights, caches, and placeholder submission templates are local.
    required = [
        ROOT / "configs" / "default.yaml",
        ROOT / "data" / "processed" / "dataset_manifest.json",
        RESULTS / "evaluation_sample.json",
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
        RESULTS / "data_stats.json",
        RESULTS / "temperature" / "temperature_results.json",
        RESULTS / "figures" / "temperature_analysis.png",
        RESULTS / "eda_summary.json",
        RESULTS / "figures" / "dataset_eda.png",
        ROOT / "scripts" / "15_eda.py",
        ROOT / "notebooks" / "01_results_walkthrough.ipynb",
        ROOT / "README.md",
        ROOT / "docs" / "report.md",
        ROOT / "docs" / "requirements_traceability.md",
        ROOT / "docs" / "publication_manifest.md",
        RESULTS / "README.md",
    ]
    checks["missing_files"] = [str(p.relative_to(ROOT)) for p in required if not p.exists()]

    sample_path = RESULTS / "evaluation_sample.json"
    sample = json.loads(sample_path.read_text(encoding="utf-8")) if sample_path.exists() else {}
    expected_ids = sample.get("test_row_indices", [])
    checks["sample_size"] = len(expected_ids)
    checks["sample_seed"] = sample.get("sample_seed")
    checks["sample_seed_42"] = sample.get("sample_seed") == 42
    checks["sample_ids_unique"] = len(expected_ids) == len(set(expected_ids))

    manifest_path = ROOT / "data" / "processed" / "dataset_manifest.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest_splits = manifest_data.get("splits", {})
    manifest_files = manifest_data.get("files", {})
    checks["sample_split_rows_match_manifest"] = (
        sample.get("test_split_rows") == manifest_splits.get("test", {}).get("pairs")
        and sample.get("test_tsv_sha256") == manifest_files.get("test", {}).get("sha256")
        and sample.get("sample_size") == len(expected_ids) == 200
    )
    notebook_path = ROOT / "notebooks" / "01_results_walkthrough.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8")) if notebook_path.exists() else {}
    notebook_text = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )
    checks["notebook_sample_scope_correct"] = (
        "same fixed 200-row sample" in notebook_text
        and "full 2,507-row test split" in notebook_text
        and "same 2,507-sentence test set" not in notebook_text
    )
    checks["notebook_embeds_eda_figure"] = "../results/figures/dataset_eda.png" in notebook_text
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    report_text = (ROOT / "docs" / "report.md").read_text(encoding="utf-8").lower()
    results_readme_text = (RESULTS / "README.md").read_text(encoding="utf-8").lower()
    checks["headline_scope_consistent"] = (
        "200-row sample drawn once from the official 2,507-row test split" in readme_text
        and "test set has 2,507 examples" in report_text
        and "sample of 200" in report_text
        and "fixed 200-row sample" in results_readme_text
        and "2,507-row test split" in results_readme_text
    )

    eda_path = RESULTS / "eda_summary.json"
    eda = json.loads(eda_path.read_text(encoding="utf-8")) if eda_path.exists() else {}
    eda_splits = eda.get("splits", {})
    saved_data_stats_path = RESULTS / "data_stats.json"
    saved_data_stats = json.loads(saved_data_stats_path.read_text(encoding="utf-8")) if saved_data_stats_path.exists() else {}
    saved_stats_splits = saved_data_stats.get("splits", {})
    eda_hashes_match = True
    eda_rows_match = True
    eda_stats_match = True
    eda_pair_quality_match = True
    eda_detail = {}
    expected_pair_quality = {
        "train": {"missing_pairs": 0, "normalized_exact_pair_duplicate_rows": 0},
        "validation": {"missing_pairs": 0, "normalized_exact_pair_duplicate_rows": 0},
        "test": {"missing_pairs": 0, "normalized_exact_pair_duplicate_rows": 1},
    }
    stat_fields = (
        ("source_characters", "src_chars"),
        ("target_characters", "tgt_chars"),
        ("source_words", "src_words"),
        ("target_words", "tgt_words"),
    )
    for split in ("train", "validation", "test"):
        item = eda_splits.get(split, {})
        source_file = item.get("source_file", {})
        expected_hash = manifest_files.get(split, {}).get("sha256")
        expected_rows = manifest_splits.get(split, {}).get("pairs")
        stats = saved_stats_splits.get(split, {})
        hashes_match = source_file.get("sha256") == expected_hash and source_file.get("path") == f"data/processed/{split}.tsv"
        rows_match = item.get("rows") == expected_rows == stats.get("pairs")
        lengths_match = all(
            all(item.get(eda_key, {}).get(key) == stats.get(stats_key, {}).get(key)
                for key in stats.get(stats_key, {}))
            for eda_key, stats_key in stat_fields
        )
        unique_match = (
            item.get("unique_source_sentences") == stats.get("unique_source_sentences")
            and item.get("unique_target_sentences") == stats.get("unique_target_sentences")
        )
        ratio_match = item.get("target_devanagari_ratio_per_sentence", {}).get("mean") == stats.get("mean_devanagari_ratio")
        split_stats_match = lengths_match and unique_match and ratio_match
        eda_hashes_match = eda_hashes_match and hashes_match
        eda_rows_match = eda_rows_match and rows_match
        eda_stats_match = eda_stats_match and split_stats_match
        pair_quality_match = all(item.get(key) == value for key, value in expected_pair_quality[split].items())
        eda_pair_quality_match = eda_pair_quality_match and pair_quality_match
        eda_detail[split] = {
            "rows": item.get("rows"),
            "manifest_rows": expected_rows,
            "source_hash_matches_manifest": hashes_match,
            "length_unique_script_stats_match": split_stats_match,
            "pair_quality_record_matches_saved_audit": pair_quality_match,
            "missing_pairs_recorded": item.get("missing_pairs"),
            "normalized_exact_pair_duplicate_rows_recorded": item.get("normalized_exact_pair_duplicate_rows"),
        }
    eda_figure = RESULTS / "figures" / "dataset_eda.png"
    eda_figure_valid = False
    if eda_figure.is_file():
        with eda_figure.open("rb") as stream:
            eda_figure_valid = eda_figure.stat().st_size > 5000 and stream.read(8) == b"\x89PNG\r\n\x1a\n"
    checks["eda"] = {
        "splits": eda_detail,
        "source_hashes_match_manifest": eda_hashes_match,
        "row_counts_match_manifest_and_data_stats": eda_rows_match,
        "distributions_match_saved_data_stats": eda_stats_match,
        "figure_valid_png": eda_figure_valid,
        "validation_scope": "checks saved EDA metadata, pair-quality record, source hashes, row counts, distributions, and figure/document references against shipped artifacts; raw TSVs are excluded, so no re-computation",
    }
    checks["eda_summary_source_hashes_match_manifest"] = eda_hashes_match
    checks["eda_rows_match_manifest"] = eda_rows_match
    checks["eda_stats_match_saved_data_stats"] = eda_stats_match
    checks["eda_pair_quality_matches_saved_audit"] = eda_pair_quality_match
    checks["eda_figure_valid"] = eda_figure_valid
    checks["eda_manifest_metadata_match"] = (
        eda.get("raw_split_rows") == manifest_data.get("raw_counts")
        and eda.get("cleaning_dropped_rows_recorded_in_manifest") == manifest_data.get("cleaning_dropped")
        and eda.get("train_duplicate_pairs_removed_recorded_in_manifest") == manifest_data.get("train_duplicates_removed")
        and eda.get("train_subset_seed") == manifest_data.get("train_subset_seed")
    )
    checks["eda_documentation_mentions_artifacts"] = all(
        path.exists() and "dataset_eda.png" in path.read_text(encoding="utf-8").lower()
        for path in (ROOT / "README.md", ROOT / "docs" / "report.md", ROOT / "docs" / "requirements_traceability.md", RESULTS / "README.md")
    )

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

    if manifest_path.exists():
        checks["split_sizes"] = {k: v.get("pairs") for k, v in manifest_splits.items()}
        checks["leakage_audit"] = manifest_data.get("leakage_audit")
        audit = manifest_data.get("leakage_audit", {})
    else:
        audit = {}

    overlap_keys = (
        "train_vs_val_source_overlap",
        "train_vs_test_source_overlap",
        "val_vs_test_source_overlap",
    )
    publication_manifest = ROOT / "docs" / "publication_manifest.md"
    publication_text = (publication_manifest.read_text(encoding="utf-8")
                        if publication_manifest.exists() else "")
    published_overlap_counts = {}
    for key in overlap_keys:
        matches = re.findall(
            rf"^\|\s*`?{re.escape(key)}`?\s*\|\s*(-?\d+)\s*\|\s*$",
            publication_text,
            flags=re.MULTILINE,
        )
        published_overlap_counts[key] = int(matches[0]) if len(matches) == 1 else None
    checks["publication_manifest_overlap_counts"] = published_overlap_counts
    checks["split_overlap_zero"] = (
        all(audit.get(key) == 0 for key in overlap_keys)
        and all(published_overlap_counts.get(key) == 0 for key in overlap_keys)
        and all(published_overlap_counts.get(key) == audit.get(key) for key in overlap_keys)
    )
    checks["overlap_gate_scope"] = (
        "validates saved overlap-audit records; raw split TSVs are excluded, so no recomputation"
    )

    temp_path = RESULTS / "temperature" / "temperature_results.json"
    temp = json.loads(temp_path.read_text(encoding="utf-8")) if temp_path.exists() else {}
    temp_results = temp.get("results", {})
    expected_temp_ids = expected_ids[:20]
    expected_temperatures = [0.2, 0.7, 1.0]
    decoder_systems = {"llama_prompt3shot", "gemma_prompt3shot"}
    temperature_coverage = {}
    for system in sorted(decoder_systems):
        records = temp_results.get(system, [])
        observed = []
        if isinstance(records, list):
            for record in records:
                value = record.get("temperature") if isinstance(record, dict) else None
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                    observed.append(float(value))
                else:
                    observed.append(None)
        temperature_coverage[system] = {
            "observed": observed,
            "exactly_once_each": (
                isinstance(records, list)
                and len(records) == len(expected_temperatures)
                and len(observed) == len(expected_temperatures)
                and all(value is not None for value in observed)
                and sorted(observed) == expected_temperatures
            ),
        }
    checks["temperature"] = {
        "n": temp.get("n"),
        "ids_match_sample_prefix": temp.get("test_row_indices") == expected_temp_ids,
        "systems": sorted(temp_results),
        "expected_temperatures": expected_temperatures,
        "coverage": temperature_coverage,
        "three_temperatures_each": all(len(v) == 3 for v in temp_results.values()),
        "exact_temperatures_once_each": all(v["exactly_once_each"] for v in temperature_coverage.values()),
    }
    checks["pass"] = (not checks["missing_files"] and checks["split_overlap_zero"]
                      and checks["sample_seed_42"]
                      and checks["sample_split_rows_match_manifest"]
                      and checks["notebook_sample_scope_correct"]
                      and checks["headline_scope_consistent"]
                      and checks["notebook_embeds_eda_figure"]
                      and checks["eda_summary_source_hashes_match_manifest"]
                      and checks["eda_rows_match_manifest"]
                      and checks["eda_stats_match_saved_data_stats"]
                      and checks["eda_pair_quality_matches_saved_audit"]
                      and checks["eda_figure_valid"]
                      and checks["eda_manifest_metadata_match"]
                      and checks["eda_documentation_mentions_artifacts"]
                      and len(expected_ids) == 200 and checks["sample_ids_unique"]
                      and all(v.get("aligned") for v in alignment.values())
                      and checks["temperature"]["n"] == 20
                      and checks["temperature"]["ids_match_sample_prefix"]
                      and checks["temperature"]["three_temperatures_each"]
                      and checks["temperature"]["exact_temperatures_once_each"]
                      and set(temp_results) == decoder_systems)
    write_json(RESULTS / "artifact_check.json", checks)
    print(json.dumps(checks, indent=2, ensure_ascii=False)[:5000])
    return 0 if checks["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
