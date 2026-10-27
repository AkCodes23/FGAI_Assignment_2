"""Fast regression tests for data integrity, tokenisation, model plumbing and metric sanity.

Run:  .venv\\Scripts\\python.exe -m pytest tests -q
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import torch

from fgai_mt.data import leak_key, load_split
from fgai_mt.metrics import corpus_scores
from fgai_mt.model_basic import TransformerMT, beam_search, count_parameters
from fgai_mt.tokenization import decode_ids, load_spm


def test_split_sizes_and_no_leakage():
    train = load_split("train")
    val = load_split("validation")
    test = load_split("test")
    assert len(train) == 60000
    assert len(val) == 520
    assert len(test) == 2507
    train_keys = {leak_key(r["en"]) for r in train}
    val_keys = {leak_key(r["en"]) for r in val}
    test_keys = {leak_key(r["en"]) for r in test}
    assert not (train_keys & val_keys), "train/validation source leakage"
    assert not (train_keys & test_keys), "train/test source leakage"
    assert not (val_keys & test_keys), "validation/test source leakage"


def test_manifest_hashes_match_files():
    manifest = json.loads((PROJECT / "data" / "processed" / "dataset_manifest.json").read_text(encoding="utf-8"))
    for split, info in manifest["files"].items():
        path = PROJECT / info["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == info["sha256"], f"{split} hash mismatch"
    assert manifest["leakage_audit"]["train_vs_test_source_overlap"] == 0


def test_tokenizer_roundtrip():
    sp = load_spm()
    text = "भारत की राजधानी नई दिल्ली है।"
    ids = sp.encode(text, out_type=int)
    assert len(ids) > 0
    assert decode_ids(sp, ids).replace(" ", "") == text.replace(" ", "")


def test_transformer_shapes_and_beam_search():
    model = TransformerMT(vocab_size=64, d_model=32, n_heads=4, n_encoder_layers=1, n_decoder_layers=1, d_ff=64)
    src = torch.tensor([[1, 5, 6, 2], [1, 7, 8, 2]])
    tgt = torch.tensor([[1, 9, 10], [1, 11, 2]])
    logits = model(src, tgt)
    assert logits.shape == (2, 3, 64)
    assert count_parameters(model) > 0
    out = beam_search(model, src, beam_size=2, max_len=8, bos_id=1, eos_id=2, pad_id=0)
    assert len(out) == 2
    assert all(isinstance(x, list) for x in out)
    assert all(len(x) <= 8 for x in out)


def test_metrics_perfect_match():
    refs = ["भारत की राजधानी नई दिल्ली है।", "यह एक परीक्षण है।"]
    scores = corpus_scores(refs, refs)
    assert scores["bleu"] > 99.9
    assert scores["chrf"] > 99.9


def test_clean_output_extraction():
    from fgai_mt.decoder import clean_output

    assert clean_output("नमस्ते दुनिया\nEnglish: hello") == "नमस्ते दुनिया"
    assert clean_output("Hindi: यह एक परीक्षण है।") == "यह एक परीक्षण है।"
    assert clean_output("<end_of_turn>") == ""


def test_prompt_ids_adds_exactly_one_bos():
    """Gemma's template starts with <bos>; TinyLlama's does not. Both must end with one."""
    from fgai_mt.decoder import prompt_ids

    class Stub:
        bos_token = "<bos>"

        def __call__(self, text, add_special_tokens=True):
            ids = [9, 8, 7]
            return {"input_ids": ([1] + ids) if add_special_tokens else ids}

    assert prompt_ids(Stub(), "<bos>translate") == [9, 8, 7]      # template already has BOS
    assert prompt_ids(Stub(), "<|system|>translate") == [1, 9, 8, 7]  # tokenizer must add it


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")

