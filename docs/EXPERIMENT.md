# Experiment

## Question

Does one epoch of QLoRA on `Qwen/Qwen2.5-Coder-7B-Instruct` improve execution accuracy on Spider dev, relative to the same model with no adapter?

## Data

Source folder: `spider_data/spider_data/`.

| Split | Files | Role |
|---|---|---|
| Train | `train_spider.json`, then `train_others.json` | Fit the adapter |
| Test | `dev.json` | Report accuracy. Never used for training |

The official Spider test split is not scored. Dev is the held-out set this experiment reports.

A row is kept only when all of these hold:

- `database/<db_id>/<db_id>.sqlite` exists
- the SQLite schema can be read
- the gold SQL executes inside 10 seconds
- the triple `(db_id, question, gold SQL)` has not already been kept

`max_train_rows` in `configs/experiment.yaml` is `null`, so every kept training row is used. Test rows are never capped.

On this run the filter kept 8,648 of 8,659 training rows (8 duplicates, 3 gold queries that failed) and all 1,034 dev rows. The schemas cover 166 databases.

Each JSONL row stores `id`, `db_id`, `db_path` (relative to the Spider root), `schema` (the `CREATE TABLE` statements), `question`, and `gold_sql`. Counts and drop reasons are written to `results/data.json`.

## Prompt

Train and test both call `text2sql.prompts.prompt_text`. The system line is fixed: "You are a SQLite expert. Return only one SQL query, no explanation." The user turn is the schema, the question, and a `SQL:` cue. The chat template of the tokenizer is applied with the generation prompt turned on.

Training loss is computed only on the gold SQL plus the end-of-sequence token. Prompt tokens are masked with `-100`. Rows whose gold SQL does not fit inside `max_seq_len` after the prompt are skipped and counted in `results/train.json`.

## Model and update

| Item | Value |
|---|---|
| Model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Load | 4-bit NF4, double quantization, bfloat16 compute |
| LoRA | r=16, alpha=32, dropout=0.05 |
| Targets | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Epochs | 1 |
| Sequence length | 1024 |
| Batch | 8, gradient accumulation 2 (effective batch 16) |
| Learning rate | 1e-4, cosine, warmup ratio 0.03 |
| Checkpoints | every 100 steps, keep 2, plus the final adapter in `checkpoints/qlora` |
| Seed | 42 |

Checkpoints are written every 100 steps. A stopped run resumes from the latest checkpoint directory.

## Test

Decoding is greedy, with at most 256 new tokens. The prediction is the text before the first semicolon, with a fenced SQL block unwrapped when the model emits one.

Primary metric: **execution accuracy**. The prediction and the gold SQL are both run on the question's SQLite file. The row is correct when the two result sets match after sorting and stringifying cells.

Secondary metrics:

- **exact match** — lowercased, stripped prediction equals the gold string. Equivalent SQL with different text counts as a miss.
- **invalid SQL** — the prediction does not execute.

`results/baseline.json` is the untuned model. `results/tuned.json` is the adapter. Both files include every scored row (`id`, gold, prediction, correct, error). `results/compare.json` records the metric delta and the ids the adapter fixed or broke.

## Measured run

Same 1,034 dev questions, greedy, before and after the adapter.

| | Before | After |
|---|---:|---:|
| Execution accuracy | 75.2% (778) | 77.1% (797) |
| Exact string match | 19.7% (204) | 46.2% (478) |
| Invalid SQL | 3.0% (31) | 4.6% (48) |
| Wrong result | 225 | 189 |

A wrong result is a query that ran and returned different rows. Invalid SQL is a query SQLite refused to run.

The adapter fixed 71 questions and broke 52. 726 were correct on both passes, and every dev id lined up (`n_unmatched` is 0).

Training used 7,940 rows. 708 kept rows were longer than 1,024 tokens and were skipped. One epoch, 490 optimizer steps, from scratch (`resumed_from` is null). Logged loss was 0.339 at step 10 and 0.045 at step 490. Peak memory was 27.6 GB on one L40S. Baseline eval took 469 seconds. Tuned eval took 492 seconds.

The France-singers question (dev ids 4 and 5) is a clear case of what the gold-SQL loss does. The gold query selects `avg(age)`, `min(age)`, `max(age)`. The untuned model selected those aggregates in a different order, so the result tables did not match. The tuned model emits the gold string, and both questions count as correct.

## What a finished run contains

`results/data.json`, `results/baseline.json`, `results/train.json`, `results/tuned.json`, and `results/compare.json`. The numbers in this file and in the README are the record of the run above. The JSON itself is gitignored and is rebuilt by the stage commands.
