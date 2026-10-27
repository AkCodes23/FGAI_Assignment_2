"""SentencePiece handling for the from-scratch Transformer (trained on the training split only)."""
from __future__ import annotations

from pathlib import Path

import sentencepiece as spm

from .config import project_path
from .data import load_split


def train_spm(vocab_size: int = 8000, model_prefix: Path | None = None, overwrite: bool = False) -> Path:
    model_prefix = Path(model_prefix or project_path("models", "spm_basic.model"))
    if model_prefix.with_suffix(".model").exists() and not overwrite:
        return model_prefix.with_suffix(".model")

    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    corpus = model_prefix.parent / "spm_train_corpus.txt"
    rows = load_split("train")
    with open(corpus, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(row["en"] + "\n")
            fh.write(row["hi"] + "\n")

    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(model_prefix.with_suffix("")),
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        byte_fallback=True,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_piece="<pad>",
        bos_piece="<s>",
        eos_piece="</s>",
        unk_piece="<unk>",
        normalization_rule_name="nmt_nfkc",
        train_extremely_large_corpus=False,
        num_threads=8,
    )
    corpus.unlink(missing_ok=True)
    return model_prefix.with_suffix(".model")


def load_spm(path: Path | None = None) -> spm.SentencePieceProcessor:
    path = Path(path or project_path("models", "spm_basic.model"))
    if not path.exists():
        path = train_spm()
    sp = spm.SentencePieceProcessor()
    sp.load(str(path))
    return sp


def decode_ids(sp: spm.SentencePieceProcessor, ids) -> str:
    """Decode token ids, dropping the four control ids (pad/bos/eos/unk)."""
    return sp.decode([int(i) for i in ids if int(i) > 3]).strip()
