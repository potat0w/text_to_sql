"""Remote stages: upload, prepare, baseline, train, test, compare."""

from __future__ import annotations

from pathlib import Path

import modal
import yaml

from text2sql.config import PACKAGE_ROOT

_RAW = yaml.safe_load((PACKAGE_ROOT / "configs" / "experiment.yaml").read_text(encoding="utf-8"))
_MODAL = _RAW["modal"]
APP_NAME = _MODAL["app_name"]
VOLUME_NAME = _MODAL["volume_name"]
GPU = _MODAL["gpu"]
ENV_NAME = _MODAL["environment"]
VOL_MOUNT = "/vol"

_SPIDER_FILES = ("train_spider.json", "train_others.json", "dev.json")


def _image_with_package(image: modal.Image) -> modal.Image:
    return image.add_local_file(
        str(PACKAGE_ROOT / "configs" / "experiment.yaml"),
        remote_path="/root/configs/experiment.yaml",
    ).add_local_python_source("text2sql")


cpu_image = _image_with_package(
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pyyaml")
    .env({"TEXT2SQL_ROOT": VOL_MOUNT})
)

gpu_image = _image_with_package(
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "torch",
        "transformers>=4.46",
        "peft>=0.13",
        "bitsandbytes>=0.44",
        "accelerate>=0.34",
        "pyyaml",
        "tqdm",
        "sentencepiece",
        "protobuf",
        "huggingface_hub",
        "hf_transfer",
    )
    .env(
        {
            "TEXT2SQL_ROOT": VOL_MOUNT,
            "HF_HOME": f"{VOL_MOUNT}/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

app = modal.App(APP_NAME, image=cpu_image)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _result_file(name: str) -> Path:
    return Path(VOL_MOUNT) / "results" / name


def _skip_if_present(name: str, force: bool) -> dict | None:
    path = _result_file(name)
    if force or not path.is_file():
        return None
    print(f"{name} is already on the volume. Skipping. Pass --force to run this stage again.")
    return {"skipped": True, "path": str(path)}


def _upload() -> None:
    spider = PACKAGE_ROOT / "spider_data" / "spider_data"
    missing = [name for name in _SPIDER_FILES if not (spider / name).is_file()]
    database = spider / "database"
    if missing or not database.is_dir():
        raise SystemExit(f"Spider source is incomplete under {spider}. Missing: {missing or ['database/']}")
    remote = "/spider_data/spider_data"
    with vol.batch_upload(force=True) as batch:
        for name in _SPIDER_FILES:
            batch.put_file(spider / name, f"{remote}/{name}")
        batch.put_directory(database, f"{remote}/database")
        for name in ("train.jsonl", "test.jsonl"):
            local = PACKAGE_ROOT / "data" / name
            if local.is_file():
                batch.put_file(local, f"/data/{name}")
        stats = PACKAGE_ROOT / "results" / "data.json"
        if stats.is_file():
            batch.put_file(stats, "/results/data.json")
    print(f"Uploaded Spider source to volume {VOLUME_NAME!r} in environment {ENV_NAME!r}.")


@app.function(image=cpu_image, volumes={VOL_MOUNT: vol}, timeout=60 * 60, cpu=2.0)
def prepare(force: bool = False) -> dict:
    from text2sql.data import build_splits

    skipped = _skip_if_present("data.json", force)
    if skipped:
        return skipped
    stats = build_splits()
    vol.commit()
    return {"train_kept": stats["train"]["kept"], "test_kept": stats["test"]["kept"]}


@app.function(image=gpu_image, gpu=GPU, volumes={VOL_MOUNT: vol}, timeout=4 * 60 * 60)
def baseline(force: bool = False) -> dict:
    from text2sql.evaluate import run_eval

    skipped = _skip_if_present("baseline.json", force)
    if skipped:
        return skipped
    summary = run_eval("baseline")
    vol.commit()
    return summary


@app.function(image=gpu_image, gpu=GPU, volumes={VOL_MOUNT: vol}, timeout=6 * 60 * 60)
def train(force: bool = False) -> dict:
    from text2sql.train import run_train

    skipped = _skip_if_present("train.json", force)
    if skipped:
        return skipped
    summary = run_train(on_checkpoint=vol.commit)
    vol.commit()
    return summary


@app.function(image=gpu_image, gpu=GPU, volumes={VOL_MOUNT: vol}, timeout=4 * 60 * 60)
def test(force: bool = False) -> dict:
    from text2sql.config import load_config
    from text2sql.evaluate import run_eval

    skipped = _skip_if_present("tuned.json", force)
    if skipped:
        return skipped
    summary = run_eval("tuned", adapter=load_config()["checkpoint_dir"])
    vol.commit()
    return summary


@app.function(image=cpu_image, volumes={VOL_MOUNT: vol}, timeout=10 * 60)
def compare(force: bool = False) -> dict:
    from text2sql.compare import run_compare

    skipped = _skip_if_present("compare.json", force)
    if skipped:
        return skipped
    summary = run_compare()
    vol.commit()
    return summary


@app.local_entrypoint()
def main(cmd: str = "", force: bool = False) -> None:
    commands = {
        "prepare": prepare,
        "baseline": baseline,
        "train": train,
        "test": test,
        "compare": compare,
    }
    if cmd == "upload":
        _upload()
        return
    if cmd not in commands:
        raise SystemExit(
            "Run one stage at a time:\n"
            "  ./modal.sh --cmd upload\n"
            "  ./modal.sh --cmd prepare\n"
            "  ./modal.sh --detach --cmd baseline\n"
            "  ./modal.sh --detach --cmd train\n"
            "  ./modal.sh --detach --cmd test\n"
            "  ./modal.sh --cmd compare\n"
            "Add --force to redo a stage that already has a result file."
        )
    print(commands[cmd].remote(force=force))
