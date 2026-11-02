from __future__ import annotations

import os
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PACKAGE_ROOT / "configs" / "experiment.yaml"


def data_root() -> Path:
    """Directory that holds spider_data/, data/, results/, and checkpoints/."""
    override = os.environ.get("TEXT2SQL_ROOT")
    if override:
        return Path(override)
    return PACKAGE_ROOT


def load_config() -> dict:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Missing config: {CONFIG_PATH}")
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    root = data_root()
    cfg = dict(raw)
    cfg["root"] = str(root)
    cfg["spider_root"] = str((root / raw["spider_root"]).resolve())
    cfg["data_dir"] = str((root / raw["data_dir"]).resolve())
    cfg["results_dir"] = str((root / raw["results_dir"]).resolve())
    cfg["checkpoint_dir"] = str((root / raw["checkpoint_dir"]).resolve())
    cfg["train_file"] = str(Path(cfg["data_dir"]) / "train.jsonl")
    cfg["test_file"] = str(Path(cfg["data_dir"]) / "test.jsonl")
    return cfg
