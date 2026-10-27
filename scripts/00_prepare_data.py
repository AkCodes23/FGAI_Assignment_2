#!/usr/bin/env python
"""Stage 0: download, clean, split and audit the English-Hindi corpus."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fgai_mt.config import dump_config, load_config  # noqa: E402
from fgai_mt.data import prepare_dataset  # noqa: E402
from fgai_mt.utils import gpu_info, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    stats = prepare_dataset(cfg)
    dump_config(cfg, Path(cfg["_project_root"]) / "logs" / "config_used_data.yaml")
    write_json(Path(cfg["_project_root"]) / "results" / "environment.json", gpu_info())
    print(json.dumps({k: stats[k] for k in ("raw_counts", "cleaning_dropped", "leakage_audit")}, indent=2, ensure_ascii=False))
    for split, s in stats["splits"].items():
        print(f"{split:10s} pairs={s['pairs']:>7} src_words_mean={s['src_words']['mean']:>6} tgt_words_mean={s['tgt_words']['mean']:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
