# MP1 — Small Language Model Challenge (HKU DASE7506)

Submission for the course protocol `7506-mp1-wt2-v2` (WikiText-2, BPE-2048,
256-token causal windows, FP32 CPU evaluation).

| Item | Value |
|---|---|
| Submitted test BPB (full test, FP32 CPU) | **1.448065** |
| Validation BPB | **1.429294** |
| Classroom baseline | 2.101260 |
| Frozen checkpoint | `checkpoints/wide512-T1-58k-wd0-s18.pt`, sha256 `e9deeb465b17b359b55aec2e3537ca53906aa5b26a2ba380f1d55c0c39e38590` |
| Configuration | `code/configs/wide512-nodrop.json` (width 512, depth 4, 16 heads, 13,634,048 parameters) |
| Budgets used | 3.8× baseline scoring time (limit 5×), 52.02 MiB assets (limit 64 MiB), 1.74 GiB peak RAM (limit 4 GiB) |
| Code (immutable) | [mp1-slm-challenge @ `mp1-final-v1`](https://github.com/cheneryang0616-bot/mp1-slm-challenge/tree/mp1-final-v1) |
| Checkpoint bundle | [mp1-checkpoint-bundle.zip](https://github.com/cheneryang0616-bot/mp1-slm-challenge/releases/download/mp1-final-v1/mp1-checkpoint-bundle.zip) — 48.31 MiB, zip sha256 `39f0d700…` |

## Evaluate the submitted score — no retraining required

```bash
cd code
python -m venv .venv && source .venv/bin/activate
python -m pip install torch==2.7.1 && python -m pip install -r requirements.txt
python -m unittest discover -s tests -v      # 5 tests, all pass
python evaluate.py --checkpoint ../checkpoints/wide512-T1-58k-wd0-s18.pt \
                   --device cpu --precision fp32 --split test
```

The printed `bpb` is the submitted score. The checkpoint sha256 is listed in
`checkpoints/MANIFEST.sha256` and in the `checkpoint_sha256` field of that run's
`metrics.json`.

## What the submission contains

| Path | Contents |
|---|---|
| `REPORT.md` / `REPORT.pdf` | the report (9 pages, limit 10) |
| `REPRODUCE.md` | exact installation, training, distillation, budget-measurement and selection procedures |
| `checkpoints/` | the frozen predictor + sha256 manifest (checkpoint bundle) |
| `code/` | code, configurations, `run_log.csv` with every run, and the course package unmodified except `train.py` (three *optional* flags) and `student.py` (ours) |
| `code/distill.py` | distils an ensemble of checkpoints into one student — the method behind the submitted model |
| `code/select_final.py` | validation-only selection + the official scorer, in one command |

The submitted model is a single ordinary checkpoint: a wide GPT trained with the
official trainer's data split and scored with the unmodified evaluator. Its
training targets come from an ensemble of 21 of our own checkpoints, averaged in
probability space and distilled into one student. `REPORT.md` §5 documents a
counter-intuitive result along the way — that the same width lever *hurts* an
under-regularised model and *helps* a properly regularised one — and §6 explains
the distillation method, its six one-variable-at-a-time levers, and three
negative results (weight averaging, log-probability averaging, temperature-scaled
distillation).

## AI assistance disclosure

Substantive AI assistance is disclosed, as the assignment requires, in
`REPORT.md` §11 and `REPRODUCE.md` §8. In summary: environment setup, the
experiment runner and queue scripts, `distill.py` / `ensemble_eval.py` /
`soup.py` / `select_final.py` / `measure_budget.py`, the implementation of the
RoPE/RMSNorm/SwiGLU block in `student.py` (verified against the provided
correctness tests), and the structure and tables of the report were
AI-assisted. Every reported number comes from a recorded run
(`code/run_log.csv` and per-run `metrics.json`); no experimental measurement was
generated or estimated by an AI system. Method choices and the analytical
argument are the author's.

No pretrained weights, external training text, retrieval index or test-set
tuning were used. Data, tokenizer, evaluator and protocol were left unchanged.
