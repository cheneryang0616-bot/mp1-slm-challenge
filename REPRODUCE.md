# Reproduction instructions

Everything below runs from the `code/` directory. Two artefacts are submitted:

| Artefact | Where | What it is for |
|---|---|---|
| **code repository** | this repository (tagged) | install, training and evaluation instructions |
| **checkpoint bundle** | separate archive | the frozen predictor; **no retraining is required** to reproduce the submitted score |

Scoring the submitted number needs **only** the checkpoint bundle (§3). Sections 4–6 describe how
that checkpoint was produced and how every number in `REPORT.md` was measured.

---

## 1. Environment

Python **3.12**, PyTorch **2.7.1**, numpy **2.5.3**, tokenizers **0.21.4**.

```bash
cd code
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
python -m pip install torch==2.7.1                  # CPU-only wheels: --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

**Verified on:** Apple Silicon (arm64), 8 cores, 17.2 GB RAM, no CUDA — training, distillation
and evaluation all run on CPU there, and the ranked evaluation is FP32 on CPU as required.
The sweeps additionally used an AMD Ryzen 7840HS and an RTX 4090 (training only; every reported
*score* was produced on CPU — see §6).

### Sanity check

```bash
python -m unittest discover -s tests -v       # 5 tests, all pass
```

Validates causality, probability normalisation, per-example independence, state reset between
windows, gradient flow, and that every target is scored exactly once.

---

## 2. Files

| Path | Origin | Note |
|---|---|---|
| `model.py`, `evaluate.py`, `common.py`, `data/`, `tests/`, `README.md`, `RUN_LOG_TEMPLATE.csv` | course-provided | **unchanged** |
| `train.py` | course-provided, **extended** | three optional flags added (`--lr`, `--weight-decay`, `--warmup`); defaults reproduce the original recipe exactly (verified bit-identical), so the documented baseline command behaves as documented |
| `student.py` | ours | the submitted model (`build_model`): pre-norm, RMSNorm, SwiGLU, RoPE, no linear biases |
| `configs/baseline.json`, `configs/modern.json`, `configs/wide*.json`, `configs/deep*.json` | ours | model configurations (`*-nodrop` = dropout 0) |
| `train.py` drivers: `experiments.py`, `run_queue.sh` | ours | experiment runner and serial queue (not needed for scoring) |
| `distill.py` | ours | distils an ensemble of checkpoints into one student |
| `ensemble_eval.py` | ours | probability-averaged ensemble scoring (also prints the log-prob average, as a diagnostic) |
| `soup.py` | ours | weight averaging — kept because it is reported as a negative result |
| `measure_budget.py`, `_peak_wrap.py`, `select_final.py` | ours | the three budget measurements, per-model peak RSS, and validation-only selection |
| `run_log.csv` | ours | every run: seed, targets, params, cost, BPB |
| `REPORT.md`, `REPRODUCE.md` | ours | the report and this file |

---

## 3. Reproduce the submitted score (no retraining)

```bash
python evaluate.py --checkpoint /path/to/<frozen-checkpoint>.pt \
                   --device cpu --precision fp32 --split test
```

This writes `test_cpu_fp32.json` next to the checkpoint and prints the **bpb** value; the
submitted score is that `bpb`.

**Frozen predictor:** checkpoint `checkpoints/wide512-T1-58k-wd0-s18.pt`, sha256
`e9deeb465b17b359b55aec2e3537ca53906aa5b26a2ba380f1d55c0c39e38590`, protocol `7506-mp1-wt2-v2`,
implementation `student`, configuration `configs/wide512-nodrop.json` (width 512, depth 4,
16 heads, 13,634,048 parameters).
The manifest in the bundle lists the sha256 of every file, and the recorded
`checkpoint_sha256` in that run's `metrics.json` must match.

---

## 4. Reproduce the pipeline that produced it

### 4.1 The teacher pool (21 checkpoints, ≈ 310 MiB)

Teachers are ordinary runs of the official trainer with `configs/wide256-dropout.json`
(width 256, depth 4, dropout 0.1, weight decay 0.3), 28,800 steps each:

```bash
for seed in 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32; do
  python train.py --implementation student --config configs/wide256-dropout.json \
                  --device cuda --precision fp32 --threads 8 --seed $seed \
                  --steps 28800 --weight-decay 0.3 --run-dir runs/wide256-24x-do-wd03-s$seed
done
```

plus five length/variant runs used as extra diversity (width 256 at 12×/16×/32×/48× with
wd 0.3, and one dropout-0.15 run). All 21 paths are listed in the `distilled_from` field of the
submitted run's `metrics.json`. On an RTX 4090 each teacher takes ≈ 15 min.

### 4.2 Check the ensemble (optional but it is the point of the method)

```bash
python ensemble_eval.py --split validation \
    --checkpoints runs/wide256-24x-do-wd03-s*/checkpoint.pt
```

Expect ≈ **1.4180** val BPB for 15 members (recorded: 1.418004) and ≈ 1.4142 for 21. The same
command prints the geometric (log-probability) average, which equals the member mean to 10
decimal places — that is the demonstration that probability averaging is the active ingredient.

### 4.3 Distil the ensemble into one student

```bash
python distill.py \
    --run-dir runs/wide512-T1-58k-wd0 \
    --config configs/wide512-nodrop.json \
    --teachers <the 21 checkpoint paths> \
    --steps 57600 --seed 17 --lr 1e-3 --weight-decay 0 --warmup 100 --hard-weight 0 \
    --temperature 1 \
    --device cuda --precision fp32 --threads 8
```

The submitted checkpoint is `--seed 18` of exactly this command; `--seed 17` was run as well and
the two were compared on **validation only** (1.429294 vs 1.430425), seed 18 being submitted.
Reported cost: 107 minutes per run on one RTX 4090.

Objectives and the exact rules are in the `distill.py` docstring:

$$p_T(v\mid x)=\tfrac{1}{M}\sum_m \operatorname{softmax}(z_m(x))_v, \qquad
\mathcal{L}=-\sum_v p_T(v\mid x)\log p_S(v\mid x)$$

**Set `--temperature 1` and keep the student's dropout at 0.** Both were measured: T = 4 is
0.043 BPB worse and a dropout-0.1 student 0.035 BPB worse. `--hard-weight 0` means pure
distillation. Soft targets are computed on **training** windows only.

Cost: ≈ 100 ms/step on a single RTX 4090, i.e. ≈ 1.7 GPU-hours for the submitted student.

### 4.4 CPU smoke test of the same code path (≈ 25 s, no GPU, no teachers required)

Uses two of the small checkpoints that the training sweeps leave behind:

```bash
python distill.py --run-dir runs/_smoke_distill_cpu --config configs/baseline.json \
    --teachers runs/verify-baseline-s17/checkpoint.pt runs/modern-s17/checkpoint.pt \
    --steps 20 --seed 17 --warmup 5 --weight-decay 0 --device cpu --precision fp32 --threads 8
```

Recorded on the reference machine: 10.5 s training, 25.0 s total. This exercises the objective,
the heterogeneous-teacher loading and the checkpoint format on ordinary hardware; it is a smoke
test, not a reproduction of the submitted score.

### 4.5 Select and score the frozen checkpoint

`select_final.py` implements the selection rule — **validation only** — and then calls the
untouched scorer once on the test split:

```bash
python select_final.py --metrics-root ../_server_results/runs \
                       --ckpt-root ../_server_results/checkpoints \
                       --split test --threads 4 \
                       wide512-T1-58k-wd0-s17 wide512-T1-58k-wd0-s18 \
                       wide448-T1-58k-wd0-s17 wide384-T1-58k-wd0-s17 wide384-T1-58k-wd0-s18
```

It prints the candidate table with validation BPB, picks the minimum, runs
`evaluate.py --device cpu --precision fp32 --split test` on it in a subprocess (so the official
measurement path is used), re-measures the baseline in the same session for the cost ratio, and
writes `runs/final_selection.json`. `--also-test-all` adds the other seeds to an appendix; those
numbers never feed the selection.

---

## 5. Measuring the three evaluation budgets

All numbers below come from one session on an idle machine, validation split, **4 CPU threads**
(the thread count of the course's own reference measurement), with the baseline re-measured in
the same session as a control and each model measured twice:

| Model | Parameters | Assets | Scoring time | × baseline | Peak RSS |
|---|---:|---:|---:|---:|---:|
| baseline | 1,088,256 | 4.17 MiB | 8.35 s (9.22 / 7.47) | 1.00 | 1.67 GiB |
| width 256 | 3,673,344 | 14.02 MiB | 14.95 s | 2.00 | 1.66 GiB |
| width 320 | 5,572,160 | 21.27 MiB | 16.99 s | 2.27 | 1.67 GiB |
| width 384 | 7,867,776 | 30.03 MiB | 20.26 s | 2.71 | 1.71 GiB |
| width 448 | 10,557,120 | 40.28 MiB | 23.28 s | 3.11 | 1.74 GiB |
| **width 512 (submitted)** | **13,634,048** | **52.02 MiB** | **27.94 s** | **3.74** | **1.77 GiB** |
| width 576 — assets over the cap | 17,110,080 | 65.28 MiB | 32.47 s | 4.34 | 1.89 GiB |
| **limit** | — | **64 MiB** | **37.4 s** | **5.00** | **4 GiB** |

Ratios are taken against the fastest baseline of the session (the conservative choice); on the
test split, in the session that produced the submitted score, the ratio is 3.77× (baseline
7.93 s, submitted 29.90 s).

How to reproduce each column:

```bash
python budget_table.py --threads 4 --split validation --repeat 2   # the whole table above
python _peak_wrap.py --checkpoint <ckpt> --device cpu --precision fp32 --threads 4 --split validation
```

- **Scoring time** — the `seconds` field that `evaluate.py` itself reports (the scorer's own
  timing), measured with the baseline in the same session so the ratio is meaningful.
- **Peak RSS** — `resource.getrusage(RUSAGE_SELF).ru_maxrss` **inside** the scoring process
  (`_peak_wrap.py`). `measure_budget.py`'s own memory column reads `RUSAGE_CHILDREN`, which is a
  high-water mark over all previous children and therefore drifts; the wrapper gives one number
  per model. macOS reports `ru_maxrss` in bytes.
- **Assets** — checkpoint file size, uncompressed. This is the binding limit at width 512
  (52.02 MiB of 64 MiB); width 576 (17.7 M parameters, 67.6 MiB) would exceed it.

**Thread sensitivity.** The ratio is *worst* at 4 threads, not at 1: width 512 costs 3.86× at
4 threads but 2.99× at 1 thread, because the baseline scales worse with thread count than wide
models do. Note also that an earlier estimate of ours put width 384 at 4.10× — that measurement
was taken with background load on the machine and is wrong; the controlled number is 2.84×.
Any published cost claim should include a baseline measured in the same session.

---

## 6. Cross-device calibration

Every device that produced a number used in the report was first made to re-run the baseline:

| Device | threads | baseline validation BPB | Δ |
|---|---:|---:|---:|
| Apple Silicon (reference) | 4 | 2.0710878209843933 | — |
| Apple Silicon, repeated 7 days later | 4 | 2.0710878209843933 | 0 (checkpoint sha256 identical) |
| AMD Ryzen 7840HS | 8 | 2.071083344398206 | −4.5 × 10⁻⁶ |
| RTX 4090 (scored on CPU) | 8 | 2.0710806613907398 | −7.2 × 10⁻⁹ |
| same config, 2 threads instead of 4 (reference machine) | 2 | 2.0710878177581518 | −3.2 × 10⁻⁹ |

All agree to 10⁻⁶, and the thread effect is six orders of magnitude below seed noise, so the
tables in `REPORT.md` mix devices and thread counts only where noted.

---

## 7. Reproducing a peer

```bash
python evaluate.py --checkpoint /path/to/peer-checkpoint.pt \
                   --device cpu --precision fp32 --split test --output peer-test.json
```

---

## 8. AI assistance disclosure

Per the assignment's rules, substantive AI assistance is disclosed here and in the report:

- **Environment setup, experiment orchestration, and the ablation / scaling / distillation
  sweeps** — AI-assisted. `experiments.py`, `run_queue.sh`, `distill.py`, `ensemble_eval.py`,
  `soup.py`, `select_final.py`, `measure_budget.py` were written with AI help; they drive the
  course-provided `train.py` / `evaluate.py` / `common.py`, which remain unmodified except for
  the three optional flags in `train.py` documented in §2.
- **`student.py`** (RoPE, RMSNorm, SwiGLU, bias-free linears) — implementation drafted with AI
  assistance; the author read, ran and verified it. Correctness is checked by the provided tests.
- **Report text** — the section skeleton and the data tables were assembled with AI assistance;
  the analytical argument is the author's. Every number comes from a recorded run.
- **Design decisions** (which component to ablate, when to stop scaling training length, when to
  switch to ensembling, when to reject the ensemble and distil instead) were made by the author
  from the measured data.

No pretrained weights, external training text, retrieval index or test-based tuning were used.
All development and model selection used the validation split only.
