"""Compare baseline and tuned result files."""

from __future__ import annotations

import json
from pathlib import Path

from text2sql.config import load_config
from text2sql.io_utils import write_json

METRIC_KEYS = ("exec_acc", "exact_match", "invalid_sql")


def run_compare(baseline_tag: str = "baseline", tuned_tag: str = "tuned") -> dict:
    cfg = load_config()
    results = Path(cfg["results_dir"])
    baseline_path = results / f"{baseline_tag}.json"
    tuned_path = results / f"{tuned_tag}.json"
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)
    if not tuned_path.is_file():
        raise FileNotFoundError(tuned_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    tuned = json.loads(tuned_path.read_text(encoding="utf-8"))

    by_id = {row["id"]: row for row in baseline["rows"]}
    fixed: list[int] = []
    broken: list[int] = []
    missing = 0
    for row in tuned["rows"]:
        before = by_id.get(row["id"])
        if before is None:
            missing += 1
            continue
        if row["correct"] and not before["correct"]:
            fixed.append(row["id"])
        elif before["correct"] and not row["correct"]:
            broken.append(row["id"])

    metrics = {}
    for key in METRIC_KEYS:
        base = float(baseline["metrics"][key])
        after = float(tuned["metrics"][key])
        metrics[key] = {"baseline": base, "tuned": after, "delta": after - base}

    summary = {
        "baseline_tag": baseline_tag,
        "tuned_tag": tuned_tag,
        "model": tuned.get("model"),
        "adapter": tuned.get("adapter"),
        "n": tuned["metrics"]["n"],
        "metrics": metrics,
        "n_fixed": len(fixed),
        "n_broken": len(broken),
        "n_unmatched": missing,
        "fixed_ids": fixed,
        "broken_ids": broken,
    }
    write_json(results / "compare.json", summary)
    for key in METRIC_KEYS:
        item = metrics[key]
        print(f"{key}: {item['baseline']:.3f} -> {item['tuned']:.3f} ({item['delta']:+.3f})")
    print(f"fixed {len(fixed)}  broken {len(broken)}")
    return {
        "metrics": metrics,
        "n_fixed": len(fixed),
        "n_broken": len(broken),
        "path": str(results / "compare.json"),
    }


def main() -> None:
    print(run_compare())


if __name__ == "__main__":
    main()
