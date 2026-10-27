"""Shared configuration and path handling for the FGAI Assignment 3 experiments."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def bootstrap_env() -> None:
    """Keep every cache inside the project directory (assignment constraint)."""
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    for sub in ("data/raw", "data/processed", "models", "outputs", "logs", "results"):
        (PROJECT_ROOT / sub).mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    bootstrap_env()
    with open(path or CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_project_root"] = str(PROJECT_ROOT)
    cfg["_config_path"] = str(path or CONFIG_PATH)
    return cfg


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def dump_config(cfg: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=True, allow_unicode=True)
