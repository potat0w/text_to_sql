"""Score data/test.jsonl by SQLite execution accuracy."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from text2sql.config import load_config
from text2sql.io_utils import read_jsonl, write_json
from text2sql.metrics import exec_match, extract_sql, resolve_db
from text2sql.modeling import load_adapter, load_base_model, load_tokenizer
from text2sql.prompts import prompt_text


def run_eval(tag: str, adapter: str | None = None) -> dict:
    cfg = load_config()
    torch.manual_seed(int(cfg["seed"]))
    test_path = Path(cfg["test_file"])
    if not test_path.is_file():
        raise FileNotFoundError(f"Missing {test_path}. Run python -m text2sql.data first.")
    rows = read_jsonl(test_path)
    if not rows:
        raise RuntimeError(f"{test_path} is empty.")

    if adapter:
        adapter_dir = Path(adapter)
        if not adapter_dir.is_dir():
            raise FileNotFoundError(f"Adapter not found: {adapter_dir}")

    tokenizer = load_tokenizer(cfg["model_name"])
    tokenizer.padding_side = "left"
    model = load_base_model(cfg["model_name"], cfg["quantization"], train_mode=False)
    if adapter:
        model = load_adapter(model, adapter)

    spider_root = Path(cfg["spider_root"])
    batch_size = int(cfg["eval"]["batch_size"])
    max_new = int(cfg["eval"]["max_new_tokens"])
    scored: list[dict] = []
    started = time.time()
    for start in tqdm(range(0, len(rows), batch_size), desc=tag):
        batch = rows[start : start + batch_size]
        prompts = [prompt_text(tokenizer, row["schema"], row["question"]) for row in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_width = encoded.input_ids.shape[1]
        for row, token_ids in zip(batch, generated):
            raw = tokenizer.decode(token_ids[prompt_width:], skip_special_tokens=True)
            prediction = extract_sql(raw)
            db_file = resolve_db(spider_root, row["db_path"])
            correct, error = exec_match(db_file, prediction, row["gold_sql"])
            scored.append(
                {
                    "id": row["id"],
                    "db_id": row["db_id"],
                    "question": row["question"],
                    "gold": row["gold_sql"],
                    "raw_output": raw,
                    "pred": prediction,
                    "correct": correct,
                    "error": error,
                }
            )

    n = len(scored)
    metrics = {
        "exec_acc": sum(item["correct"] for item in scored) / n,
        "exact_match": sum(item["pred"].strip().lower() == item["gold"].strip().lower() for item in scored) / n,
        "invalid_sql": sum(item["error"] == "invalid_sql" for item in scored) / n,
        "n": n,
    }
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    payload = {
        "tag": tag,
        "model": cfg["model_name"],
        "adapter": adapter,
        "metrics": metrics,
        "eval_seconds": round(time.time() - started, 1),
        "gpu": gpu_name,
        "rows": scored,
    }
    out = Path(cfg["results_dir"]) / f"{tag}.json"
    write_json(out, payload)
    print(metrics)
    return {"tag": tag, "metrics": metrics, "eval_seconds": payload["eval_seconds"], "gpu": gpu_name, "path": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--adapter", default=None)
    args = parser.parse_args()
    print(run_eval(args.tag, args.adapter))


if __name__ == "__main__":
    main()
