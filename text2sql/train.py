"""QLoRA fine-tune. Loss is on the gold SQL only."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

from text2sql.config import load_config
from text2sql.io_utils import read_jsonl, write_json
from text2sql.modeling import compute_dtype, load_base_model, load_tokenizer
from text2sql.prompts import prompt_text


class _Encoded(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def _encode(tokenizer, row: dict, max_seq_len: int) -> dict | None:
    prompt_ids = tokenizer(
        prompt_text(tokenizer, row["schema"], row["question"]),
        add_special_tokens=False,
    ).input_ids
    answer_ids = tokenizer(
        row["gold_sql"].strip() + tokenizer.eos_token,
        add_special_tokens=False,
    ).input_ids
    input_ids = (prompt_ids + answer_ids)[:max_seq_len]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_seq_len]
    if not any(label != -100 for label in labels):
        return None
    return {"input_ids": input_ids, "labels": labels}


def _collate(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collate(batch: list[dict]) -> dict:
        width = max(len(row["input_ids"]) for row in batch)
        input_ids, attention, labels = [], [], []
        for row in batch:
            pad = width - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [pad_id] * pad)
            attention.append([1] * len(row["input_ids"]) + [0] * pad)
            labels.append(row["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention),
            "labels": torch.tensor(labels),
        }

    return collate


def _run_meta(train_cfg: dict) -> dict:
    return {
        "batch_size": int(train_cfg["batch_size"]),
        "grad_accum": int(train_cfg["grad_accum"]),
        "max_seq_len": int(train_cfg["max_seq_len"]),
        "epochs": train_cfg["epochs"],
        "lr": train_cfg["lr"],
    }


def _fresh_if_settings_changed(output_dir: Path, train_cfg: dict) -> str | None:
    """Drop checkpoints from a different batch setup so training starts clean."""
    meta = _run_meta(train_cfg)
    meta_path = output_dir / "run_meta.json"
    saved = json.loads(meta_path.read_text()) if meta_path.is_file() else None
    if saved != meta:
        for path in output_dir.glob("checkpoint-*"):
            if path.is_dir():
                shutil.rmtree(path)
        meta_path.write_text(json.dumps(meta) + "\n")
        print("Starting from scratch for the current batch settings.")
        return None
    return _latest_checkpoint(output_dir)


def _latest_checkpoint(output_dir: Path) -> str | None:
    found: list[tuple[int, Path]] = []
    if not output_dir.is_dir():
        return None
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        step = path.name.rsplit("-", 1)[-1]
        if step.isdigit():
            found.append((int(step), path))
    if not found:
        return None
    return str(max(found)[1])


def run_train(on_checkpoint: Callable[[], None] | None = None) -> dict:
    cfg = load_config()
    quant, lora, train_cfg = cfg["quantization"], cfg["lora"], cfg["training"]
    set_seed(int(cfg["seed"]))
    output_dir = Path(cfg["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(cfg["results_dir"])

    train_path = Path(cfg["train_file"])
    if not train_path.is_file():
        raise FileNotFoundError(f"Missing {train_path}. Run python -m text2sql.data first.")

    tokenizer = load_tokenizer(cfg["model_name"])
    encoded: list[dict] = []
    skipped_long = 0
    for row in read_jsonl(train_path):
        item = _encode(tokenizer, row, int(train_cfg["max_seq_len"]))
        if item is None:
            skipped_long += 1
            continue
        encoded.append(item)
    if not encoded:
        raise RuntimeError("No training rows left after sequence-length filtering.")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    dtype = compute_dtype(quant)
    model = load_base_model(cfg["model_name"], quant, train_mode=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora["r"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            target_modules=list(lora["target_modules"]),
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    callbacks = []
    if on_checkpoint is not None:

        class _CommitCallback(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                on_checkpoint()

        callbacks.append(_CommitCallback())

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(train_cfg["epochs"]),
        learning_rate=float(train_cfg["lr"]),
        per_device_train_batch_size=int(train_cfg["batch_size"]),
        gradient_accumulation_steps=int(train_cfg["grad_accum"]),
        # Transformers 5 takes the warmup fraction as a float warmup_steps.
        warmup_steps=float(train_cfg["warmup_ratio"]),
        lr_scheduler_type=train_cfg["lr_scheduler"],
        weight_decay=float(train_cfg["weight_decay"]),
        logging_steps=int(train_cfg["logging_steps"]),
        save_strategy="steps",
        save_steps=int(train_cfg["save_steps"]),
        save_total_limit=int(train_cfg["save_total_limit"]),
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        optim="paged_adamw_8bit" if quant["load_in_4bit"] else "adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=int(cfg["seed"]),
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=_Encoded(encoded),
        data_collator=_collate(tokenizer),
        callbacks=callbacks,
    )
    resume = _fresh_if_settings_changed(output_dir, train_cfg)
    if resume:
        print("Resuming from", resume)
    trainer.train(resume_from_checkpoint=resume)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logs = [row for row in trainer.state.log_history if "loss" in row]
    peak = None
    gpu_name = "cpu"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    summary = {
        "model": cfg["model_name"],
        "train_rows_file": len(read_jsonl(train_path)),
        "train_rows_used": len(encoded),
        "skipped_longer_than_max_seq_len": skipped_long,
        "epochs": train_cfg["epochs"],
        "final_loss": logs[-1]["loss"] if logs else None,
        "resumed_from": resume,
        "checkpoint_dir": str(output_dir),
        "gpu": gpu_name,
        "peak_gpu_mem_gb": peak,
        "loss_history": [
            {"step": row.get("step"), "epoch": row.get("epoch"), "loss": row["loss"]} for row in logs
        ],
    }
    write_json(results_dir / "train.json", summary)
    print("adapter saved to", output_dir)
    return {
        "train_rows_used": summary["train_rows_used"],
        "skipped_longer_than_max_seq_len": skipped_long,
        "final_loss": summary["final_loss"],
        "gpu": gpu_name,
        "peak_gpu_mem_gb": peak,
        "checkpoint_dir": str(output_dir),
    }


def main() -> None:
    print(run_train())


if __name__ == "__main__":
    main()
