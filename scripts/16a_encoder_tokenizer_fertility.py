#!/usr/bin/env python
"""Measure encoder-decoder tokenizer fertility on the local full test split.

This performs tokenization only; it does not load model weights or run inference.
The default processed TSV is local-only and must match the published SHA-256 in
data/processed/dataset_manifest.json. The basic SentencePiece model is included
in results/tokenizers; mT5 uses the pinned public tokenizer revision below.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import sentencepiece as spm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
TEST_SHA256 = "c4b0e3f255f1ffc1f56eb6afdb64220f9e521c6ebf68e63c48f2919a025e8e75"
MT5_ID = "google/mt5-small"
MT5_REVISION = "73fb5dbe4756edadc8fbe8c769b0a109493acf7a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_hindi_rows(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Processed test TSV is local-only and is not in the publication set: {path}"
        )
    actual = sha256(path)
    if actual != TEST_SHA256:
        raise ValueError(f"Test TSV SHA-256 mismatch: expected {TEST_SHA256}, got {actual}")
    with path.open(encoding="utf-8") as stream:
        if stream.readline().rstrip("\n").split("\t") != ["en", "hi"]:
            raise ValueError("Expected processed TSV columns in the order en, hi")
        rows = []
        for line_number, line in enumerate(stream, start=2):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2 or not fields[1]:
                raise ValueError(f"Malformed processed test row at line {line_number}")
            rows.append(fields)
    if len(rows) != 2507:
        raise ValueError(f"Expected 2,507 test rows with Hindi references, got {len(rows)}")
    return [row[1] for row in rows]


def summarize(counts: list[int], references: list[str]) -> dict[str, float | int]:
    words = [len(text.split()) for text in references]
    return {
        "n_sentences": len(counts),
        "hi_tokens_mean": round(statistics.fmean(counts), 2),
        "hi_tokens_median": statistics.median(counts),
        "hi_tokens_p95": sorted(counts)[int(0.95 * len(counts)) - 1],
        "hi_words_mean": round(statistics.fmean(words), 2),
        "hi_tokens_per_word": round(statistics.fmean(counts) / statistics.fmean(words), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-tsv", type=Path, default=ROOT / "data/processed/test.tsv")
    parser.add_argument("--basic-model", type=Path, default=ROOT / "results/tokenizers/spm_basic.model")
    parser.add_argument("--output", type=Path, default=ROOT / "results/tokenizer_fertility.json")
    parser.add_argument(
        "--mt5-tokenizer-source",
        default=f"{MT5_ID}@{MT5_REVISION}",
        help="Override the pinned Hub tokenizer with a local directory for offline use.",
    )
    args = parser.parse_args()

    references = load_hindi_rows(args.test_tsv.resolve())
    if not args.basic_model.is_file():
        raise FileNotFoundError(f"Basic SentencePiece model not found: {args.basic_model}")
    processor = spm.SentencePieceProcessor(model_file=str(args.basic_model.resolve()))
    basic_counts = [len(processor.encode(text, out_type=int)) for text in references]

    source = args.mt5_tokenizer_source
    if source == f"{MT5_ID}@{MT5_REVISION}":
        tokenizer = AutoTokenizer.from_pretrained(MT5_ID, revision=MT5_REVISION, use_fast=False)
        recorded_source = {"model_id": MT5_ID, "revision": MT5_REVISION}
    else:
        tokenizer = AutoTokenizer.from_pretrained(source, use_fast=False, local_files_only=True)
        recorded_source = {"local_path": str(Path(source).resolve())}
    mt5_counts = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in references]

    out_path = args.output.resolve()
    existing = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    existing["basic_transformer_sentencepiece"] = summarize(basic_counts, references)
    existing[MT5_ID] = summarize(mt5_counts, references)
    existing["_measurement"] = {
        "method": "Hindi reference subword count; no special tokens; mean tokens divided by mean whitespace words",
        "test_tsv_sha256": TEST_SHA256,
        "test_tsv": "data/processed/test.tsv (local-only; excluded from publication)",
        "test_rows": len(references),
        "basic_model_sha256": sha256(args.basic_model.resolve()),
        "mt5_tokenizer": recorded_source,
        "inference_or_training": False,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: existing[key] for key in ("basic_transformer_sentencepiece", MT5_ID, "_measurement")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
