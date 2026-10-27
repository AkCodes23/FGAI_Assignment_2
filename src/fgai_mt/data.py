"""Dataset acquisition, cleaning, splitting and provenance for English -> Hindi MT.

Only the public IIT Bombay English-Hindi corpus (CC BY-NC 4.0) is used. The provider's
official train/validation/test splits are preserved; a fixed-size training subset is drawn
from the official train split and shared by every system.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import PROJECT_ROOT, bootstrap_env
from .utils import write_json

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
WHITESPACE = re.compile(r"\s+")
CONTROL = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]")
PUNCT = re.compile(r"[^\w\s\u0900-\u097F]", flags=re.UNICODE)

SPLIT_FILES = {
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_raw(hf_id: str) -> dict[str, Path]:
    """Download the three official parquet shards into the project-local Hugging Face cache."""
    bootstrap_env()
    from huggingface_hub import hf_hub_download

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for split, filename in SPLIT_FILES.items():
        path = hf_hub_download(repo_id=hf_id, filename=filename, repo_type="dataset")
        out[split] = Path(path)
    return out


def _records(path: Path) -> Iterable[tuple[str, str]]:
    df = pd.read_parquet(path)
    if "translation" in df.columns:
        for item in df["translation"]:
            yield str(item.get("en", "")), str(item.get("hi", ""))
    elif {"en", "hi"}.issubset(df.columns):
        for en, hi in zip(df["en"], df["hi"]):
            yield str(en), str(hi)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unrecognised schema in {path}: {df.columns.tolist()}")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    text = CONTROL.sub(" ", text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return WHITESPACE.sub(" ", text).strip()


def devanagari_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if DEVANAGARI.match(c)) / len(letters)


def clean_pair(en: str, hi: str, cfg: dict[str, Any]) -> tuple[str, str] | None:
    en, hi = normalise(en), normalise(hi)
    if len(en) < cfg["min_chars"] or len(hi) < cfg["min_chars"]:
        return None
    if en.casefold() == hi.casefold():
        return None
    ratio = max(len(en), len(hi)) / max(1, min(len(en), len(hi)))
    if ratio > cfg["max_char_ratio"]:
        return None
    if devanagari_ratio(hi) < 0.2:  # target must actually be Hindi
        return None
    if len(en.split()) > 120 or len(hi.split()) > 120:
        return None
    return en, hi


def leak_key(text: str) -> str:
    """Aggressive normalisation used only for leakage/dedup comparison."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = PUNCT.sub(" ", text)
    return WHITESPACE.sub(" ", text).strip()


def _dedupe(pairs: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], int]:
    seen: set[str] = set()
    kept: list[tuple[str, str]] = []
    dropped = 0
    for en, hi in pairs:
        key = leak_key(en) + "\t" + leak_key(hi)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append((en, hi))
    return kept, dropped


def _length_stats(rows: list[tuple[str, str]]) -> dict[str, Any]:
    import numpy as np

    src_chars = np.array([len(en) for en, _ in rows])
    tgt_chars = np.array([len(hi) for _, hi in rows])
    src_words = np.array([len(en.split()) for en, _ in rows])
    tgt_words = np.array([len(hi.split()) for _, hi in rows])

    def q(arr):
        return {f"p{p}": round(float(np.percentile(arr, p)), 1) for p in (50, 90, 95, 99)} | {
            "mean": round(float(arr.mean()), 1),
            "max": int(arr.max()),
        }

    return {
        "pairs": len(rows),
        "src_chars": q(src_chars),
        "tgt_chars": q(tgt_chars),
        "src_words": q(src_words),
        "tgt_words": q(tgt_words),
        "unique_source_sentences": len({leak_key(en) for en, _ in rows}),
        "unique_target_sentences": len({leak_key(hi) for _, hi in rows}),
        "mean_devanagari_ratio": round(float(np.mean([devanagari_ratio(hi) for _, hi in rows])), 4),
    }


def prepare_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    dcfg = cfg["dataset"]
    bootstrap_env()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw = download_raw(dcfg["hf_id"])

    raw_counts: dict[str, int] = {}
    cleaned: dict[str, list[tuple[str, str]]] = {}
    dropped: dict[str, int] = {}
    for split, path in raw.items():
        rows = list(_records(path))
        raw_counts[split] = len(rows)
        kept = [pair for en, hi in rows if (pair := clean_pair(en, hi, dcfg)) is not None]
        dropped[split] = len(rows) - len(kept)
        cleaned[split] = kept

    # ---- leakage / dedup -------------------------------------------------
    val_test_keys = {leak_key(en) for en, _ in cleaned["validation"]} | {
        leak_key(en) for en, _ in cleaned["test"]
    }
    val_test_full = {
        leak_key(en) + "\t" + leak_key(hi)
        for en, hi in cleaned["validation"] + cleaned["test"]
    }

    train_dedup, dup_train = _dedupe(cleaned["train"])
    train_no_leak = [
        (en, hi)
        for en, hi in train_dedup
        if leak_key(en) not in val_test_keys
        and leak_key(en) + "\t" + leak_key(hi) not in val_test_full
    ]
    leakage_removed = len(train_dedup) - len(train_no_leak)

    # ---- fixed shared training subset -----------------------------------
    import random

    rng = random.Random(cfg["seed"])
    n_subset = min(int(dcfg["train_subset"]), len(train_no_leak))
    train_subset = rng.sample(train_no_leak, n_subset)
    rng.shuffle(train_subset)

    splits = {"train": train_subset, "validation": cleaned["validation"], "test": cleaned["test"]}

    # ---- persist ---------------------------------------------------------
    paths = {}
    for split, rows in splits.items():
        path = PROCESSED_DIR / f"{split}.tsv"
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write("en\thi\n")
            for en, hi in rows:
                fh.write(f"{en}\t{hi}\n")
        paths[split] = path

    stats = {
        "dataset": {
            "name": dcfg["name"],
            "hf_id": dcfg["hf_id"],
            "homepage": dcfg["homepage"],
            "license": dcfg["license"],
            "citation": dcfg["citation"],
        },
        "raw_counts": raw_counts,
        "cleaning_dropped": dropped,
        "train_duplicates_removed": dup_train,
        "train_rows_removed_as_val_test_leakage": leakage_removed,
        "splits": {
            split: _length_stats(rows) for split, rows in splits.items()
        },
        "train_subset_seed": cfg["seed"],
        "source_links": {
            "hf_mirror": f"https://huggingface.co/datasets/{dcfg['hf_id']}",
            "official": dcfg["homepage"],
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": {split: {"path": str(p.relative_to(PROJECT_ROOT)), "sha256": _sha256(p)} for split, p in paths.items()},
    }

    # ---- extra leakage evidence -----------------------------------------
    def keys(rows):
        return Counter(leak_key(en) for en, _ in rows)

    stats["leakage_audit"] = {
        "train_vs_val_source_overlap": len(set(keys(splits["train"])) & set(keys(splits["validation"]))),
        "train_vs_test_source_overlap": len(set(keys(splits["train"])) & set(keys(splits["test"]))),
        "val_vs_test_source_overlap": len(set(keys(splits["validation"])) & set(keys(splits["test"]))),
        "method": "Unicode NFKC + casefold + punctuation-strip + whitespace-fold; exact source sentence match required after normalisation",
    }

    write_json(PROCESSED_DIR / "dataset_manifest.json", stats)
    write_json(PROJECT_ROOT / "results" / "data_stats.json", stats)
    return stats


def load_split(split: str) -> list[dict[str, str]]:
    path = PROCESSED_DIR / f"{split}.tsv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run scripts/00_prepare_data.py first")
    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        assert header == ["en", "hi"], header
        for line in fh:
            en, hi = line.rstrip("\n").split("\t")
            rows.append({"en": en, "hi": hi})
    return rows
