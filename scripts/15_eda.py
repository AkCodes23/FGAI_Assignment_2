#!/usr/bin/env python
"""Create reproducible descriptive EDA from the saved IITB processed splits."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MANIFEST_PATH = PROCESSED / "dataset_manifest.json"
DATA_STATS_PATH = RESULTS / "data_stats.json"
SUMMARY_PATH = RESULTS / "eda_summary.json"
FIGURE_PATH = FIGURES / "dataset_eda.png"
SPLITS = ("train", "validation", "test")
WHITESPACE = re.compile(r"\s+")
PUNCT = re.compile(r"[^\w\s\u0900-\u097F]", flags=re.UNICODE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def leak_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return WHITESPACE.sub(" ", PUNCT.sub(" ", text)).strip()


def devanagari_ratio(text: str) -> float:
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return 0.0
    return sum("\u0900" <= character <= "\u097f" for character in letters) / len(letters)


def distribution(values: list[int | float], decimals: int = 1) -> dict[str, int | float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "p10": round(float(np.percentile(arr, 10)), decimals),
        "p50": round(float(np.percentile(arr, 50)), decimals),
        "p90": round(float(np.percentile(arr, 90)), decimals),
        "p95": round(float(np.percentile(arr, 95)), decimals),
        "p99": round(float(np.percentile(arr, 99)), decimals),
        "mean": round(float(np.mean(arr)), decimals),
        "max": int(np.max(arr)),
    }


def script_class(character: str) -> str:
    if "\u0900" <= character <= "\u097f":
        return "devanagari"
    if "LATIN" in unicodedata.name(character, ""):
        return "latin"
    return "other"


def summarize_split(name: str, expected_file: dict[str, Any], saved_stats: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    path = PROCESSED / f"{name}.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing processed split: {path}")
    actual_hash = sha256(path)
    expected_hash = expected_file.get("sha256")
    if actual_hash != expected_hash:
        raise ValueError(f"SHA-256 mismatch for {path.name}: {actual_hash} != {expected_hash}")

    src_chars: list[int] = []
    tgt_chars: list[int] = []
    src_words: list[int] = []
    tgt_words: list[int] = []
    target_dev_ratio: list[float] = []
    source_keys: set[str] = set()
    target_keys: set[str] = set()
    pair_keys: set[tuple[str, str]] = set()
    duplicate_pair_rows = 0
    missing_pairs = 0
    target_script_letters = {"devanagari": 0, "latin": 0, "other": 0}

    with path.open("r", encoding="utf-8", newline="") as stream:
        if stream.readline().rstrip("\r\n") != "en\thi":
            raise ValueError(f"Unexpected TSV header in {path.name}")
        rows = 0
        for line_number, line in enumerate(stream, start=2):
            line = line.rstrip("\r\n")
            if "\t" not in line:
                raise ValueError(f"Missing source/target delimiter in {path.name}:{line_number}")
            # This project's TSV writer stores one normalized pair per physical
            # line and does not CSV-escape quotation marks. Split only on the
            # first tab so punctuation quotes remain ordinary text characters.
            source, target = line.split("\t", 1)
            rows += 1
            if not source.strip() or not target.strip():
                missing_pairs += 1
            source_key, target_key = leak_key(source), leak_key(target)
            pair_key = (source_key, target_key)
            if pair_key in pair_keys:
                duplicate_pair_rows += 1
            pair_keys.add(pair_key)
            source_keys.add(source_key)
            target_keys.add(target_key)
            src_chars.append(len(source))
            tgt_chars.append(len(target))
            src_words.append(len(source.split()))
            tgt_words.append(len(target.split()))
            target_dev_ratio.append(devanagari_ratio(target))
            for character in target:
                if character.isalpha():
                    target_script_letters[script_class(character)] += 1

    if rows == 0:
        raise ValueError(f"No rows found in {path.name}")
    letter_total = sum(target_script_letters.values())
    target_script_share = {
        key: round(value / letter_total, 6) if letter_total else 0.0
        for key, value in target_script_letters.items()
    }
    computed = {
        "pairs": rows,
        "src_chars": distribution(src_chars),
        "tgt_chars": distribution(tgt_chars),
        "src_words": distribution(src_words),
        "tgt_words": distribution(tgt_words),
        "unique_source_sentences": len(source_keys),
        "unique_target_sentences": len(target_keys),
        "mean_devanagari_ratio": round(float(np.mean(target_dev_ratio)), 4),
    }

    # Confirm that this independent pass reproduces the already-saved dataset
    # statistics. The gate can then validate this summary without raw TSVs.
    distribution_fields = ("src_chars", "tgt_chars", "src_words", "tgt_words")
    stats_match = computed["pairs"] == saved_stats.get("pairs")
    stats_match = stats_match and all(
        all(computed[field][key] == saved_stats.get(field, {}).get(key) for key in saved_stats.get(field, {}))
        for field in distribution_fields
    )
    stats_match = stats_match and all(
        computed[field] == saved_stats.get(field)
        for field in ("unique_source_sentences", "unique_target_sentences", "mean_devanagari_ratio")
    )
    summary = {
        "rows": rows,
        "missing_pairs": missing_pairs,
        "normalized_exact_pair_duplicate_rows": duplicate_pair_rows,
        "unique_source_sentences": len(source_keys),
        "unique_target_sentences": len(target_keys),
        "source_characters": computed["src_chars"],
        "target_characters": computed["tgt_chars"],
        "source_words": computed["src_words"],
        "target_words": computed["tgt_words"],
        "target_devanagari_ratio_per_sentence": distribution(target_dev_ratio, decimals=4),
        "target_alphabetic_script_letter_counts": target_script_letters,
        "target_alphabetic_script_letter_shares": target_script_share,
        "matches_saved_data_stats": stats_match,
        "source_file": {"path": f"data/processed/{name}.tsv", "sha256": actual_hash},
    }
    plot_data = {
        "source_words": src_words,
        "target_words": tgt_words,
        "source_characters": src_chars,
        "target_characters": tgt_chars,
        "target_dev_ratio": target_dev_ratio,
        "script_shares": target_script_share,
    }
    return summary, plot_data


def make_figure(summaries: dict[str, dict[str, Any]], plot_data: dict[str, dict[str, list[Any]]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    split_names = list(SPLITS)
    split_labels = {"train": "Train (60k subset)", "validation": "Validation", "test": "Test"}
    colors = {"train": "#2f6b8a", "validation": "#dd8452", "test": "#55a868"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.3))
    for ax, metric, title, ylabel in (
        (axes[0, 0], "words", "Sentence length: whitespace tokens", "Words per sentence"),
        (axes[0, 1], "characters", "Sentence length: Unicode code points", "Characters per sentence"),
    ):
        src_key, tgt_key = ("source_words", "target_words") if metric == "words" else ("source_characters", "target_characters")
        groups, labels, group_colors = [], [], []
        for split in split_names:
            groups.extend((plot_data[split][src_key], plot_data[split][tgt_key]))
            labels.extend((f"{split_labels[split]}\nsource", f"{split_labels[split]}\ntarget"))
            group_colors.extend((colors[split], colors[split]))
        boxes = ax.boxplot(groups, patch_artist=True, showfliers=False, widths=0.62)
        for patch, color in zip(boxes["boxes"], group_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
        ax.set_xticks(range(1, len(labels) + 1), labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    bins = np.linspace(0, 1, 21)
    for split in split_names:
        ax.hist(plot_data[split]["target_dev_ratio"], bins=bins, density=True, histtype="step", linewidth=1.8,
                color=colors[split], label=split_labels[split])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Devanagari letters / target alphabetic letters")
    ax.set_ylabel("Density")
    ax.set_title("Hindi-target script share per sentence")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    classes = ("devanagari", "latin", "other")
    bottoms = np.zeros(len(split_names))
    script_colors = {"devanagari": "#4c9f70", "latin": "#e09f3e", "other": "#7b6d8d"}
    for script in classes:
        shares = [plot_data[split]["script_shares"][script] for split in split_names]
        ax.bar([split_labels[split] for split in split_names], shares, bottom=bottoms, color=script_colors[script], label=script.title())
        bottoms += np.asarray(shares)
    for split in split_names:
        shares = plot_data[split]["script_shares"]
        latin_other_pct = 100 * (shares["latin"] + shares["other"])
        ax.text(split_labels[split], 1.015, f"Latin + other: {latin_other_pct:.2f}%",
                ha="center", va="bottom", fontsize=7, color="#333333")
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("Share of alphabetic target characters")
    ax.set_title("Hindi target alphabetic-character scripts")
    ax.tick_params(axis="x", labelrotation=12)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("IIT Bombay English–Hindi processed split EDA", fontsize=15, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    saved_stats = json.loads(DATA_STATS_PATH.read_text(encoding="utf-8"))
    expected_files = manifest.get("files", {})
    stats_splits = saved_stats.get("splits", {})
    summaries: dict[str, dict[str, Any]] = {}
    plot_data: dict[str, dict[str, list[Any]]] = {}
    for split in SPLITS:
        summary, values = summarize_split(split, expected_files.get(split, {}), stats_splits.get(split, {}))
        if not summary["matches_saved_data_stats"]:
            raise ValueError(f"Recomputed {split} statistics do not match results/data_stats.json")
        summaries[split] = summary
        plot_data[split] = values

    eda_summary = {
        "schema_version": 1,
        "dataset": manifest["dataset"],
        "raw_split_rows": manifest.get("raw_counts", {}),
        "cleaning_dropped_rows_recorded_in_manifest": manifest.get("cleaning_dropped", {}),
        "train_duplicate_pairs_removed_recorded_in_manifest": manifest.get("train_duplicates_removed"),
        "train_subset_seed": manifest.get("train_subset_seed"),
        "method": {
            "processed_training_scope": "fixed 60,000-row shared training subset",
            "validation_and_test_scope": "full provider validation and test splits",
            "headline_evaluation_scope": "separate fixed 200-row sample from the 2,507-row test split",
            "character_length": "Python Unicode code-point count",
            "word_length": "whitespace-separated token count",
            "unique_sentence_and_pair_key": "same NFKC, casefold, punctuation-strip, whitespace-fold key as src/fgai_mt/data.py",
            "target_devanagari_ratio": "Devanagari code points divided by alphabetic target characters, per sentence",
            "target_script_composition": "alphabetic target characters classified by Unicode script name",
        },
        "splits": summaries,
        "interpretation_limitations": [
            "This is descriptive corpus EDA, not a human translation-quality or representativeness audit.",
            "The training statistics describe the saved 60,000-row subset, not the full raw training corpus.",
            "The fixed 200-row headline sample is distinct from the full 2,507-row test-split EDA.",
            "Repeated source or target sentences can remain after exact normalized pair deduplication.",
        ],
        "figure": "results/figures/dataset_eda.png",
    }
    make_figure(summaries, plot_data)
    SUMMARY_PATH.write_text(json.dumps(eda_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": "results/eda_summary.json",
        "figure": "results/figures/dataset_eda.png",
        "split_rows": {name: value["rows"] for name, value in summaries.items()},
        "missing_pairs": {name: value["missing_pairs"] for name, value in summaries.items()},
        "normalized_exact_pair_duplicate_rows": {name: value["normalized_exact_pair_duplicate_rows"] for name, value in summaries.items()},
        "data_stats_match": {name: value["matches_saved_data_stats"] for name, value in summaries.items()},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
