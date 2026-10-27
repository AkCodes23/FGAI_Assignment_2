"""Fixed, reproducible held-out sample shared by every model evaluation."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .config import load_config, project_path
from .data import load_split
from .utils import write_json


SAMPLE_PATH = project_path("results", "evaluation_sample.json")


def load_eval_rows(limit: int | None = None) -> list[tuple[int, dict[str, str]]]:
    """Return fixed test rows with their original row indices; create the manifest once."""
    test = load_split("test")
    test_path = project_path("data", "processed", "test.tsv")
    test_hash = hashlib.sha256(test_path.read_bytes()).hexdigest()
    cfg = load_config()
    settings = cfg["evaluation"]
    seed = int(settings.get("sample_seed", cfg["seed"]))
    sample_size = min(int(settings.get("test_size", 200)), len(test))

    if SAMPLE_PATH.exists():
        sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
        if sample.get("test_tsv_sha256") != test_hash:
            raise RuntimeError("evaluation_sample.json was created for a different processed test split")
        indices = [int(i) for i in sample["test_row_indices"]]
        if len(indices) != int(sample["sample_size"]) or len(set(indices)) != len(indices):
            raise RuntimeError("evaluation sample manifest has duplicate or inconsistent row IDs")
        if any(i < 0 or i >= len(test) for i in indices):
            raise RuntimeError("evaluation sample contains an out-of-range test row ID")
    else:
        indices = sorted(random.Random(seed).sample(range(len(test)), sample_size))
        sample = {
            "selection": "uniform random sample without replacement; sorted by original test order",
            "sample_seed": seed,
            "sample_size": sample_size,
            "test_split_rows": len(test),
            "test_tsv_sha256": test_hash,
            "test_row_indices": indices,
        }
        write_json(SAMPLE_PATH, sample)

    if limit is not None:
        indices = indices[:limit]
    return [(i, test[i]) for i in indices]
