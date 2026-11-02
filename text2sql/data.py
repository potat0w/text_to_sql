"""Build train.jsonl and test.jsonl from Spider.

Train is train_spider.json plus train_others.json. Test is dev.json.
A row is kept when its SQLite file exists, the schema can be read, and the gold SQL executes.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from text2sql.config import load_config
from text2sql.io_utils import write_json, write_jsonl
from text2sql.metrics import run_sql

TRAIN_FILES = ("train_spider.json", "train_others.json")
DEV_FILE = "dev.json"


def _schema(db_file: Path) -> str:
    connection = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL "
            "ORDER BY name"
        ).fetchall()
    finally:
        connection.close()
    return "\n".join(row[0].strip() for row in rows if row[0])


def _find_sqlite(spider_root: Path, db_id: str) -> Path | None:
    candidate = spider_root / "database" / db_id / f"{db_id}.sqlite"
    if candidate.is_file():
        return candidate
    return None


def _load_examples(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _convert(examples: list[dict], spider_root: Path, schemas: dict[str, str], start_id: int) -> tuple[list[dict], dict[str, int], int]:
    kept: list[dict] = []
    dropped = {"missing_db": 0, "no_schema": 0, "gold_error": 0, "duplicate": 0}
    seen: set[tuple[str, str, str]] = set()
    next_id = start_id
    for index, example in enumerate(examples, start=1):
        if index % 2000 == 0:
            print(f"  checked {index}/{len(examples)}", flush=True)
        db_id = example["db_id"]
        question = example["question"].strip()
        gold = example["query"].strip()
        key = (db_id, question, gold)
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        db_file = _find_sqlite(spider_root, db_id)
        if db_file is None:
            dropped["missing_db"] += 1
            continue
        if db_id not in schemas:
            try:
                schemas[db_id] = _schema(db_file)
            except Exception:
                schemas[db_id] = ""
        schema = schemas[db_id]
        if not schema.strip():
            dropped["no_schema"] += 1
            continue
        try:
            run_sql(db_file, gold, timeout_s=10.0)
        except Exception:
            dropped["gold_error"] += 1
            continue
        kept.append(
            {
                "id": next_id,
                "db_id": db_id,
                "db_path": db_file.relative_to(spider_root).as_posix(),
                "schema": schema,
                "question": question,
                "gold_sql": gold,
            }
        )
        next_id += 1
    return kept, dropped, next_id


def _preview(rows: list[dict], limit: int) -> list[dict]:
    return [
        {"id": row["id"], "db_id": row["db_id"], "question": row["question"], "gold_sql": row["gold_sql"]}
        for row in rows[:limit]
    ]


def build_splits() -> dict:
    cfg = load_config()
    spider_root = Path(cfg["spider_root"])
    if not (spider_root / "train_spider.json").is_file():
        found = sorted(p.name for p in spider_root.iterdir()) if spider_root.is_dir() else []
        raise FileNotFoundError(
            f"train_spider.json not found under {spider_root}. "
            f"Contents: {found or 'directory missing'}."
        )
    dev_path = spider_root / DEV_FILE
    if not dev_path.is_file():
        raise FileNotFoundError(dev_path)

    train_paths = [spider_root / name for name in TRAIN_FILES if (spider_root / name).is_file()]
    raw_train = _load_examples(train_paths)
    raw_dev = _load_examples([dev_path])
    schemas: dict[str, str] = {}
    train_rows, train_dropped, _ = _convert(raw_train, spider_root, schemas, 0)
    test_rows, test_dropped, _ = _convert(raw_dev, spider_root, schemas, 0)

    cap = cfg.get("max_train_rows")
    train_before_cap = len(train_rows)
    if isinstance(cap, int) and cap >= 0:
        train_rows = train_rows[:cap]
        for index, row in enumerate(train_rows):
            row["id"] = index

    data_dir = Path(cfg["data_dir"])
    write_jsonl(data_dir / "train.jsonl", train_rows)
    write_jsonl(data_dir / "test.jsonl", test_rows)

    stats = {
        "spider_root": str(spider_root),
        "train_sources": [path.name for path in train_paths],
        "test_source": DEV_FILE,
        "train": {
            "raw": len(raw_train),
            "kept_before_cap": train_before_cap,
            "kept": len(train_rows),
            "max_train_rows": cap,
            "dropped": train_dropped,
        },
        "test": {"raw": len(raw_dev), "kept": len(test_rows), "dropped": test_dropped},
        "n_schemas": len(schemas),
        "examples": {"train": _preview(train_rows, 3), "test": _preview(test_rows, 2)},
    }
    write_json(Path(cfg["results_dir"]) / "data.json", stats)
    print(
        json.dumps(
            {
                "train_kept": len(train_rows),
                "test_kept": len(test_rows),
                "train_file": cfg["train_file"],
                "test_file": cfg["test_file"],
            },
            indent=2,
        )
    )
    return stats


def main() -> None:
    build_splits()


if __name__ == "__main__":
    main()
