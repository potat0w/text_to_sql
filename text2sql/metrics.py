from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path


def extract_sql(text: str) -> str:
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        text = fenced.group(1)
    return text.strip().split(";")[0].strip()


def resolve_db(spider_root: Path, db_path: str) -> Path:
    candidate = Path(db_path)
    if candidate.is_file():
        return candidate
    nested = spider_root / db_path
    if nested.is_file():
        return nested
    raise FileNotFoundError(nested)


def run_sql(db_file: Path, sql: str, timeout_s: float = 10.0) -> list:
    connection = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    connection.text_factory = lambda raw: raw.decode("utf-8", errors="ignore")
    started = time.monotonic()

    def _interrupt() -> int:
        return 1 if time.monotonic() - started > timeout_s else 0

    connection.set_progress_handler(_interrupt, 10000)
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


def exec_match(db_file: Path, prediction: str, gold: str) -> tuple[bool, str | None]:
    """Return (correct, error). error is None on a match."""
    try:
        gold_rows = run_sql(db_file, gold)
    except Exception:
        return False, "gold_error"
    try:
        pred_rows = run_sql(db_file, prediction)
    except Exception:
        return False, "invalid_sql"
    key = lambda rows: sorted(tuple(map(str, row)) for row in rows)
    if key(pred_rows) == key(gold_rows):
        return True, None
    return False, "wrong_result"
