#!/usr/bin/env python
"""Create the shared random test sample manifest before model comparison."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fgai_mt.eval_sample import SAMPLE_PATH, load_eval_rows  # noqa: E402


if __name__ == "__main__":
    rows = load_eval_rows()
    sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    print(f"Saved {len(rows)} fixed test rows to {SAMPLE_PATH}")
    print(f"seed={sample['sample_seed']} test_sha256={sample['test_tsv_sha256']}")
