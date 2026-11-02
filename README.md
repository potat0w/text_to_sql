# It only counts if the SQL runs

Fine-tune a 7B coder model on [Spider](https://yale-lily.github.io/spider), then grade it the way a database would. The predicted query is executed. The gold query is executed. The answer is correct when the two result sets match.

The same prompt, the same 1,034 dev questions, and the same checker score the model before and after one epoch of QLoRA. Dev is never used for training.

## This run

Spider dev, greedy decoding, one question at a time. Counts are out of 1,034.

| | Before | After one epoch |
|---|---:|---:|
| Execution accuracy | 75.2% (778) | 77.1% (797) |
| Exact string match | 19.7% (204) | 46.2% (478) |
| Invalid SQL | 3.0% (31) | 4.6% (48) |

Execution accuracy is the score that matters here: both queries ran, and the rows matched. Exact match only checks the text. Two different strings can return the same rows, and two similar strings can return different ones.

Question by question, the adapter fixed 71 and broke 52. 726 were already correct and stayed correct.

Training saw 7,940 of the 8,648 kept training rows. The other 708 were longer than 1,024 tokens, so they were left out. Logged loss went from 0.339 at step 10 to 0.045 at step 490. The run used one L40S and peaked at 27.6 GB. Each eval pass took about eight minutes.

## What the training is learning

The loss is on the gold SQL only. Prompt tokens are masked, so the adapter is trained to continue the prompt with Spider's SQL, not with an explanation.

That is why exact match moves as far as it does. Before training, the model often wrote a query with the same intent and different wording. After one epoch, it much more often wrote the gold string.

One dev question: "What is the average, minimum, and maximum age of all singers from France?"

The gold query asks for `avg`, then `min`, then `max`. Before training, the model emitted `min`, then `max`, then `avg`. The columns came back in a different order, so the result sets did not match. After training, the prediction is the gold string, and the rows match.

`results/compare.json` names every question that changed. The row-level files are rebuilt by the commands below and are not committed.

## Setup

| | |
|---|---|
| Model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Update | 4-bit QLoRA, rank 16, one epoch |
| Loss | Gold SQL tokens only. The prompt is masked. |
| Train | `train_spider` + `train_others`. A row is kept only if its database exists and the gold SQL runs. |
| Test | Spider dev, 1,034 questions |
| Seed | 42 |

Every knob is in `configs/experiment.yaml`. The protocol, the filters, and the full counts are in `docs/EXPERIMENT.md`.

## Layout

```
text2sql/data.py        Spider JSON  ->  train.jsonl, test.jsonl
text2sql/prompts.py     one prompt, shared by train and test
text2sql/train.py       QLoRA adapter
text2sql/evaluate.py    execution accuracy
text2sql/compare.py     fixed vs broken
text2sql/modal_app.py   one remote stage per command
```

## Recreate

Spider is about 1 GB of SQLite databases, so it is not in git. Download the release from [yale-lily.github.io/spider](https://yale-lily.github.io/spider) and place it so these paths exist:

```
spider_data/spider_data/train_spider.json
spider_data/spider_data/train_others.json
spider_data/spider_data/dev.json
spider_data/spider_data/database/
```

Then:

```bash
pip install -r requirements.txt
pip install modal
python -m text2sql.data
```

`python -m text2sql.data` writes the JSONL files and `results/data.json`. The GPU stages need the [Modal](https://modal.com) CLI, authenticated, and an environment named `text2sql` (the name is in `configs/experiment.yaml`). `./modal.sh` reads a gitignored `.profile` when one is present, so a local profile override stays off the remote.

GPU work is one stage at a time. A failed stage leaves the earlier result files in place.

```bash
./modal.sh --cmd upload
./modal.sh --cmd prepare
./modal.sh --detach --cmd baseline
./modal.sh --detach --cmd train
./modal.sh --detach --cmd test
./modal.sh --cmd compare
./modal.sh --cmd pull
```

`prepare` and `compare` use CPU. `baseline`, `train`, and `test` each use one GPU, and only while that command is running. A stage that already wrote its result file does nothing. Add `--force` to run it again. Training resumes from the latest checkpoint until `results/train.json` exists.
