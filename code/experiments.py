"""Token-budget-matched experiment runner for MP1.

Every comparison here keeps the number of processed training targets equal to
the classroom baseline (1200 steps x 32 seqs x 256 targets = 9,830,400), so a
difference in BPB can be attributed to the mechanism rather than to more data.

Plan entries only choose `--implementation` and `--config`, so the official
`train.py` / `evaluate.py` stay untouched.  A mechanism ablation is therefore
just another config file that switches one component off.

Usage
-----
  python experiments.py list
  python experiments.py run <plan> [--seeds 17 18] [--steps N] [--threads 4]
                                  [--batch-size 32] [--test] [--force] [--dry-run]
  python experiments.py status
  python experiments.py log            # rebuild runs/run_log.csv from run dirs

Finished runs are skipped automatically, so re-running a plan resumes it.
Test scoring is off by default: use `--test` only once the method is frozen.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    import resource
    import resource as _resource  # noqa: F401
except ImportError:      # Windows has no `resource` module (POSIX only).
    resource = None

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / '.venv/bin/python'
if not PYTHON.exists():
    PYTHON = ROOT / '.venv/Scripts/python.exe'   # Windows venv layout
PYTHON = str(PYTHON) if PYTHON.exists() else sys.executable

BASELINE_TARGETS = 1200 * 32 * 256          # 9,830,400 processed targets
CONTEXT = 256
HARDWARE = f'{platform.machine()} CPU / {platform.platform()}'

# ---------------------------------------------------------------------------
# The experiment plan.  Add an entry here to register a new configuration.
#   run_prefix  : run directories are named  runs/<run_prefix>-s<seed>
#   seeds       : all seeds to run; the spread across them is the noise floor
#   note        : free text recorded in the run log
# ---------------------------------------------------------------------------
PLANS = {
    'baseline-seeds': dict(
        implementation='model', config='configs/baseline.json', run_prefix='baseline',
        seeds=[17, 18, 19], batch_size=32,
        note='seed-noise estimate for the classroom baseline'),
    'modern': dict(
        implementation='student', config='configs/modern.json', run_prefix='modern',
        seeds=[17, 18, 19], batch_size=32,
        note='main method: RMSNorm + SwiGLU + RoPE + no bias'),
    'ablate-swiglu': dict(
        implementation='student', config='configs/modern-gelu.json', run_prefix='ablate-swiglu',
        seeds=[17, 18, 19], batch_size=32,
        note='ablation: GELU MLP in place of SwiGLU, everything else identical'),
    'ablate-rope': dict(
        implementation='student', config='configs/modern-pos.json', run_prefix='ablate-rope',
        seeds=[17, 18, 19], batch_size=32,
        note='ablation: learned absolute positions in place of RoPE'),
    'ablate-norm': dict(
        implementation='student', config='configs/modern-layernorm.json', run_prefix='ablate-norm',
        seeds=[17, 18, 19], batch_size=32,
        note='ablation: LayerNorm in place of RMSNorm'),
    # Training length is not restricted by the assignment -- only inference is
    # budgeted -- so this is the cheapest remaining lever. `targets` overrides
    # the baseline-matched budget for plans that deliberately train longer.
    'modern-long': dict(
        implementation='student', config='configs/modern.json', run_prefix='modern-long',
        seeds=[17], batch_size=32, targets=3 * BASELINE_TARGETS,
        note='same architecture as modern, 3x training targets'),
    'modern-6x': dict(
        implementation='student', config='configs/modern.json', run_prefix='modern-6x',
        seeds=[17], batch_size=32, targets=6 * BASELINE_TARGETS,
        note='same architecture as modern, 6x training targets'),
    'modern-9x': dict(
        implementation='student', config='configs/modern.json', run_prefix='modern-9x',
        seeds=[17], batch_size=32, targets=9 * BASELINE_TARGETS,
        note='same architecture as modern, 9x training targets'),
    # Scaling experiment: same recipe, ~2.06x the parameters (width 128 -> 192,
    # heads 4 -> 6, head_dim kept at 32). Trained at the 3x budget, since the
    # training-length curve has saturated and the eval budget still has room.
    'wide192': dict(
        implementation='student', config='configs/wide192.json', run_prefix='wide192',
        seeds=[17], batch_size=32, targets=3 * BASELINE_TARGETS,
        note='modern recipe at 2.06x parameters, 3x training targets'),
    'wide256': dict(
        implementation='student', config='configs/wide256.json', run_prefix='wide256',
        seeds=[17], batch_size=32, targets=3 * BASELINE_TARGETS,
        note='modern recipe at 3.5x parameters, 3x training targets'),
    'wide320': dict(
        implementation='student', config='configs/wide320.json', run_prefix='wide320',
        seeds=[17], batch_size=32, targets=3 * BASELINE_TARGETS,
        note='modern recipe at 5.3x parameters, 3x training targets'),
    # Combines the two levers: the largest size that still had a clear win, trained
    # longer. Motivated by the under-training argument -- at 3x budget the 3.67M-parameter
    # model sees only ~0.4x the tokens-per-parameter that scaling laws suggest.
    'wide256-6x': dict(
        implementation='student', config='configs/wide256.json', run_prefix='wide256-6x',
        seeds=[17], batch_size=32, targets=6 * BASELINE_TARGETS,
        note='3.5x parameters AND 6x training targets'),
    # The second cell of the 2x2 size-vs-training grid. Compared against
    # wide256-6x at the same budget: if width 320 wins here, the plateau seen at
    # the 3x budget was an under-training artefact rather than a data ceiling.
    'wide320-6x': dict(
        implementation='student', config='configs/wide320.json', run_prefix='wide320-6x',
        seeds=[17], batch_size=32, targets=6 * BASELINE_TARGETS,
        note='5.3x parameters AND 6x training targets'),
    'wide256-9x': dict(
        implementation='student', config='configs/wide256.json', run_prefix='wide256-9x',
        seeds=[17], batch_size=32, targets=9 * BASELINE_TARGETS,
        note='3.5x parameters at 9x training targets, to extend the 256 curve'),
    # The 3x -> 6x result at width 256 was a clear regression (+0.052 BPB, 13 sigma),
    # i.e. overfitting on the repeated corpus. Probe just above the known optimum.
    'wide256-4.5x': dict(
        implementation='student', config='configs/wide256.json', run_prefix='wide256-4.5x',
        seeds=[17], batch_size=32, targets=9 * BASELINE_TARGETS // 2,
        note='3.5x parameters at 4.5x training targets, probing the peak of the curve'),
    # Same size and budget as wide256-6x, but with dropout. Single variable change,
    # so it isolates the effect of regularisation under demonstrated overfitting.
    'wide256-6x-do': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='wide256-6x-do',
        seeds=[17], batch_size=32, targets=6 * BASELINE_TARGETS,
        note='width 256 at 6x targets with dropout 0.1, against demonstrated overfitting'),
    # -----------------------------------------------------------------------
    # Hyper-parameter sweep on the best known configuration (width 256, 3x
    # budget, so ~52 min per run). The trainer's lr / weight decay / warmup were
    # hard-coded and had never been varied; these four runs bracket the original
    # values (lr 1e-3, weight decay 0.1) on both sides, one knob at a time.
    # -----------------------------------------------------------------------
    'hp-lr-small': dict(
        implementation='student', config='configs/wide256.json', run_prefix='hp-lr-small',
        seeds=[17], batch_size=32, targets=3 * BASELINE_TARGETS, lr=5e-4,
        note='HP sweep: peak lr 5e-4 (reference 1e-3), everything else unchanged'),
    'hp-lr-large': dict(
        implementation='student', config='configs/wide256.json', run_prefix='hp-lr-large',
        seeds=[17], batch_size=32, targets=3 * BASELINE_TARGETS, lr=2e-3,
        note='HP sweep: peak lr 2e-3 (reference 1e-3), everything else unchanged'),
    'hp-wd-zero': dict(
        implementation='student', config='configs/wide256.json', run_prefix='hp-wd-zero',
        seeds=[17], batch_size=32, targets=3 * BASELINE_TARGETS, weight_decay=0.0,
        note='HP sweep: weight decay 0 (reference 0.1), everything else unchanged'),
    'hp-wd-large': dict(
        implementation='student', config='configs/wide256.json', run_prefix='hp-wd-large',
        seeds=[17], batch_size=32, targets=3 * BASELINE_TARGETS, weight_decay=0.3,
        note='HP sweep: weight decay 0.3 (reference 0.1), everything else unchanged'),
    # -----------------------------------------------------------------------
    # Ensemble / multi-seed confirmation of the best configuration (width 256,
    # 3x budget).  Seeds 18 and 19 complete the 3-seed set for wide256, which
    # serves two purposes: a proper mean +- sd for the best config, and members
    # for a log-probability ensemble (ensemble_eval.py) or a weight-average soup
    # (soup.py).  The inference budget is only about a third used, so there is
    # room to spend it on an ensemble -- and unlike a bigger model, ensembling
    # cannot overfit the repeated corpus.
    # -----------------------------------------------------------------------
    'wide256-seeds': dict(
        implementation='student', config='configs/wide256.json', run_prefix='wide256',
        seeds=[18, 19], batch_size=32, targets=3 * BASELINE_TARGETS,
        note='extra seeds for the best config: 3-seed stats + ensemble/soup members'),
    # -----------------------------------------------------------------------
    # Depth axis.  Width was pushed to its plateau at ~3.5M parameters, but
    # depth was never varied at all.  These two configs hold the parameter
    # budget roughly fixed (3.54M / 3.49M vs wide256's 3.67M) and trade width
    # for depth, so any difference is attributable to the *shape* of the model
    # rather than to its size.
    # -----------------------------------------------------------------------
    'deep224x5': dict(
        implementation='student', config='configs/deep224x5.json', run_prefix='deep224x5',
        seeds=[17], batch_size=32, targets=3 * BASELINE_TARGETS,
        note='depth axis: width 224 x depth 5 (3.47M params, same budget as wide256)'),
    'deep192x7': dict(
        implementation='student', config='configs/deep192x7.json', run_prefix='deep192x7',
        seeds=[17], batch_size=32, targets=3 * BASELINE_TARGETS,
        note='depth axis: width 192 x depth 7 (3.49M params, same budget as wide256)'),
    # -----------------------------------------------------------------------
    # Follow-ups motivated by the hyper-parameter sweep: weight decay was the
    # only knob that moved the needle (0 -> 0.1 -> 0.3 got monotonically
    # better), which fits the project's main story -- the bottleneck is
    # overfitting on a repeated corpus, so regularisation is what helps.
    #
    # `wide256-wd03` gives the new best config (width 256, 3x budget,
    # weight decay 0.3) a proper 3-seed mean +- sd.  Seed 17 already exists as
    # run `hp-wd-large-s17` and is mirrored into runs/wide256-wd03-s17 by hand
    # (same command, same seed) so the three seeds share one run prefix.
    # -----------------------------------------------------------------------
    'wide256-wd03': dict(
        implementation='student', config='configs/wide256.json', run_prefix='wide256-wd03',
        seeds=[18, 19], batch_size=32, targets=3 * BASELINE_TARGETS, weight_decay=0.3,
        note='seed confirmation of the best recipe so far: 3x budget, weight decay 0.3'),
    # The 3x -> 6x regression (+0.052) was measured under weight decay 0.1.  If
    # stronger decay absorbs the overfitting, more training may pay off again --
    # this is the only route left to beat ~1.62.
    'wide256-6x-wd03': dict(
        implementation='student', config='configs/wide256.json', run_prefix='wide256-6x-wd03',
        seeds=[17], batch_size=32, targets=6 * BASELINE_TARGETS, weight_decay=0.3,
        note='6x budget with weight decay 0.3: does regularisation rescue the 6x regression?'),
    # Is 0.3 the peak, or does the curve keep going up?
    'hp-wd-xlarge': dict(
        implementation='student', config='configs/wide256.json', run_prefix='hp-wd-xlarge',
        seeds=[17], batch_size=32, targets=3 * BASELINE_TARGETS, weight_decay=0.5,
        note='HP sweep: weight decay 0.5 (reference 0.1, previous best 0.3)'),
    # Two regularisers at once, at the 3x budget where nothing else moved.
    'wide256-wd03-do': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='wide256-wd03-do',
        seeds=[17], batch_size=32, targets=3 * BASELINE_TARGETS, weight_decay=0.3,
        note='weight decay 0.3 AND dropout 0.1 at the 3x budget: two regularisers stacked'),
    # -----------------------------------------------------------------------
    # Dropout reopened both axes.  Adding dropout 0.1 at the 6x budget took
    # validation BPB from 1.693554 (the old 6x regression) to 1.574134, so the
    # "3x is optimal" and "more parameters hurt" conclusions were both
    # artefacts of overfitting.  These four runs chase that.
    #
    # They are listed in priority order; 1 and 2 are the high-value ones.
    # -----------------------------------------------------------------------
    'wide256-9x-do': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='wide256-9x-do',
        seeds=[17], batch_size=32, targets=9 * BASELINE_TARGETS,
        note='training-length axis under dropout: 9x budget, dropout 0.1 (6x+do already won)'),
    'wide320-6x-do': dict(
        implementation='student', config='configs/wide320-dropout.json', run_prefix='wide320-6x-do',
        seeds=[17], batch_size=32, targets=6 * BASELINE_TARGETS,
        note='size axis under dropout: width 320 at 6x, dropout 0.1'),
    'wide256-6x-do-wd03': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='wide256-6x-do-wd03',
        seeds=[17], batch_size=32, targets=6 * BASELINE_TARGETS, weight_decay=0.3,
        note='stack two regularisers: dropout 0.1 AND weight decay 0.3 at 6x'),
    'wide256-6x-do-s18': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='wide256-6x-do',
        seeds=[18], batch_size=32, targets=6 * BASELINE_TARGETS,
        note='second seed of the current best recipe (6x + dropout 0.1) for freeze-time stats'),
    # -----------------------------------------------------------------------
    # Cross-device calibration.  Run this FIRST on a new machine: it reproduces
    # the classroom baseline from scratch, and seed 17 is known to give
    # validation BPB 2.071088 on the original CPU box.  The gap tells you how
    # much the new hardware/threads move the numbers before you trust any
    # comparison against runs made elsewhere.  1200 steps: a few minutes.
    # -----------------------------------------------------------------------
    'verify-baseline': dict(
        implementation='model', config='configs/baseline.json', run_prefix='verify-baseline',
        seeds=[17], batch_size=32,
        note='cross-device calibration: must reproduce 2.071088 (CPU, seed 17)'),
    # -----------------------------------------------------------------------
    # Thread-count control.  Everything here is CPU-only, and torch parallelises
    # matmul reductions, so the summation *order* -- and therefore the exact
    # weights -- depend on the thread count.  The baseline seeds and `modern`
    # happened to be trained with 4 threads while every later run used 2, so the
    # ablation table silently compares across two settings.  These two runs
    # measure that effect directly on the classroom baseline, whose 4-thread
    # seed-17 value is known (2.071088), and double as a determinism check:
    # same seed + same thread count should reproduce the recorded number.
    # -----------------------------------------------------------------------
    'threads-4t': dict(
        implementation='model', config='configs/baseline.json', run_prefix='threads-4t',
        seeds=[17], batch_size=32,
        note='thread control: baseline seed 17 at 4 threads (reproducibility check)'),
    'threads-2t': dict(
        implementation='model', config='configs/baseline.json', run_prefix='threads-2t',
        seeds=[17], batch_size=32,
        note='thread control: baseline seed 17 at 2 threads (direct 2T vs 4T comparison)'),
    # NOT NEEDED (2026-09-23): the thread effect was measured directly --
    # `threads-2t` vs `threads-4t` on the same seed differs by 3.2e-9 BPB, six
    # orders of magnitude below the 0.004 seed noise, while a same-thread rerun
    # reproduces the original checkpoint's sha256 exactly.  Kept only so the
    # ablation table *could* be put on one protocol if a reviewer asks.
    'modern-2t': dict(
        implementation='student', config='configs/modern.json', run_prefix='modern-2t',
        seeds=[17, 18, 19], batch_size=32,
        note='modern re-run at 2 threads, to match the ablation protocol'),
    # -----------------------------------------------------------------------
    # Round 3: the training-length axis is still open.  9x + dropout reached
    # validation 1.550847 on the remote Ryzen box (calibrated against the local
    # 2.071088, delta -4.5e-6), which is another -0.0233 beyond 6x + dropout.
    # So: find the peak of the length curve, and stack the other confirmed
    # lever (weight decay 0.3) on top of it.
    # -----------------------------------------------------------------------
    'wide256-12x-do': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='wide256-12x-do',
        seeds=[17], batch_size=32, targets=12 * BASELINE_TARGETS,
        note='length axis: 12x + dropout 0.1, to locate the peak above 9x'),
    'wide256-9x-do-wd03': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='wide256-9x-do-wd03',
        seeds=[17], batch_size=32, targets=9 * BASELINE_TARGETS, weight_decay=0.3,
        note='stack both confirmed levers: 9x + dropout 0.1 + weight decay 0.3'),
    'wide256-9x-do-s18': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='wide256-9x-do',
        seeds=[18], batch_size=32, targets=9 * BASELINE_TARGETS,
        note='second seed of the 9x + dropout recipe (needed for freeze-time stats)'),
    'wide320-9x-do': dict(
        implementation='student', config='configs/wide320-dropout.json', run_prefix='wide320-9x-do',
        seeds=[17], batch_size=32, targets=9 * BASELINE_TARGETS,
        note='size axis under dropout: width 320 at 9x, dropout 0.1'),
    # -----------------------------------------------------------------------
    # Round 4 -- for a GPU / many-core server, where a run costs minutes rather
    # than hours.  A GPU is ~50x the local 2-thread Mac, so the strategy changes
    # from "pick one probe" to "sweep the axis".
    #
    # `gpu-anchor-*` are the cross-device bridges and MUST run first: they repeat
    # two known runs so the deltas can be checked against the recorded values
    # (1.574134 and 1.550847), which is a much stronger check than the baseline
    # calibration alone.
    # -----------------------------------------------------------------------
    'gpu-anchor-6x-do': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='gpu-anchor-6x-do',
        seeds=[17], batch_size=32, targets=6 * BASELINE_TARGETS,
        note='cross-device bridge: repeat of wide256-6x-do (recorded 1.574134)'),
    'gpu-anchor-9x-do': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='gpu-anchor-9x-do',
        seeds=[17], batch_size=32, targets=9 * BASELINE_TARGETS,
        note='cross-device bridge: repeat of wide256-9x-do (recorded 1.550847)'),
    'wide256-16x-do': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='wide256-16x-do',
        seeds=[17], batch_size=32, targets=16 * BASELINE_TARGETS,
        note='length axis: 16x + dropout 0.1, above the 12x probe'),
    'wide256-24x-do': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='wide256-24x-do',
        seeds=[17], batch_size=32, targets=24 * BASELINE_TARGETS,
        note='length axis: 24x + dropout 0.1, to bracket the peak'),
    'wide256-16x-do-wd03': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='wide256-16x-do-wd03',
        seeds=[17], batch_size=32, targets=16 * BASELINE_TARGETS, weight_decay=0.3,
        note='stack both levers at the long end: 16x + dropout 0.1 + weight decay 0.3'),
    # The size axis was only ever closed under wd 0.1 with no dropout, where
    # bigger models overfitted.  With dropout the ceiling moves back up -- but it
    # is capped by the *inference* budget (~10-15M parameters: 64 MiB of assets
    # and 5x the baseline CPU scoring time), so do not go past width 512.
    'wide320-12x-do': dict(
        implementation='student', config='configs/wide320-dropout.json', run_prefix='wide320-12x-do',
        seeds=[17], batch_size=32, targets=12 * BASELINE_TARGETS,
        note='size axis revisited: width 320 (5.57M) at 12x + dropout 0.1'),
    # -----------------------------------------------------------------------
    # Round 5: dense sweeps of the three axes that still matter, made cheap by
    # the GPU (~30 ms/step vs 1.16 s on the 2-thread Mac).  Listed in the order
    # they should run: most valuable first, because the instance may be shut
    # down on a clock.
    # -----------------------------------------------------------------------
    # (a) training length -- fill the 16x-24x gap and bracket the peak above 24x
    'wide256-20x-do': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='wide256-20x-do',
        seeds=[17], batch_size=32, targets=20 * BASELINE_TARGETS,
        note='length axis: 20x + dropout 0.1 (fills the 16x-24x gap)'),
    'wide256-32x-do': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='wide256-32x-do',
        seeds=[17], batch_size=32, targets=32 * BASELINE_TARGETS,
        note='length axis: 32x + dropout 0.1 (brackets the peak from above)'),
    # (b) size axis at a FIXED length (12x): 256 -> 320 -> 384 -> 448.  Capped at
    #     448 by the inference budget: 10.55M params = 42 MB of assets and ~28 s
    #     of CPU scoring, both inside the 64 MiB / 5x-baseline limits.
    'wide384-12x-do': dict(
        implementation='student', config='configs/wide384-dropout.json', run_prefix='wide384-12x-do',
        seeds=[17], batch_size=32, targets=12 * BASELINE_TARGETS,
        note='size axis at 12x: width 384 (7.87M) + dropout 0.1'),
    'wide448-12x-do': dict(
        implementation='student', config='configs/wide448-dropout.json', run_prefix='wide448-12x-do',
        seeds=[17], batch_size=32, targets=12 * BASELINE_TARGETS,
        note='size axis at 12x: width 448 (10.55M) + dropout 0.1, near the asset budget'),
    # (c) dropout *magnitude* -- 0.1 was an arbitrary first guess and dropout is
    #     the single most powerful knob in the project, so its value matters.
    'wide256-9x-do15': dict(
        implementation='student', config='configs/wide256-dropout15.json', run_prefix='wide256-9x-do15',
        seeds=[17], batch_size=32, targets=9 * BASELINE_TARGETS,
        note='dropout sweep at 9x: dropout 0.15'),
    'wide256-9x-do20': dict(
        implementation='student', config='configs/wide256-dropout20.json', run_prefix='wide256-9x-do20',
        seeds=[17], batch_size=32, targets=9 * BASELINE_TARGETS,
        note='dropout sweep at 9x: dropout 0.2'),
    # (d) weight decay upper end, at the long-training end
    'wide256-9x-wd05': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='wide256-9x-wd05',
        seeds=[17], batch_size=32, targets=9 * BASELINE_TARGETS, weight_decay=0.5,
        note='weight-decay sweep at 9x: weight decay 0.5'),
    'wide256-24x-do-wd03': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='wide256-24x-do-wd03',
        seeds=[17], batch_size=32, targets=24 * BASELINE_TARGETS, weight_decay=0.3,
        note='stack both levers at the long end: 24x + dropout 0.1 + weight decay 0.3'),
    # -----------------------------------------------------------------------
    # Round 6: fill the rest of the night.  The length curve has converged
    # (6->9: -0.0233, 9->12: -0.0095, 12->16: -0.0048 -- halving each step), so
    # the remaining upside is in the *other* axes, and above all in getting a
    # proper noise floor AT THE OPERATING POINT: every sigma this project has was
    # measured at the 3x budget, and sub-0.005 deltas at 16x cannot be judged
    # without it.  The seed runs double as ensemble/soup members.
    # -----------------------------------------------------------------------
    'wide256-16x-do-s18': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='wide256-16x-do',
        seeds=[18], batch_size=32, targets=16 * BASELINE_TARGETS,
        note='seed + ensemble member of 16x + dropout 0.1 (noise floor at the operating point)'),
    'wide256-16x-do-s19': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='wide256-16x-do',
        seeds=[19], batch_size=32, targets=16 * BASELINE_TARGETS,
        note='seed + ensemble member of 16x + dropout 0.1 (noise floor at the operating point)'),
    'wide256-16x-do-s20': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='wide256-16x-do',
        seeds=[20], batch_size=32, targets=16 * BASELINE_TARGETS,
        note='seed + ensemble member of 16x + dropout 0.1 (noise floor at the operating point)'),
    # size axis at the best length found so far (16x): 256 -> 320 -> 384 -> 448
    'wide320-16x-do': dict(
        implementation='student', config='configs/wide320-dropout.json', run_prefix='wide320-16x-do',
        seeds=[17], batch_size=32, targets=16 * BASELINE_TARGETS,
        note='size x length: width 320 (5.57M) at 16x + dropout 0.1'),
    'wide384-16x-do': dict(
        implementation='student', config='configs/wide384-dropout.json', run_prefix='wide384-16x-do',
        seeds=[17], batch_size=32, targets=16 * BASELINE_TARGETS,
        note='size x length: width 384 (7.87M) at 16x + dropout 0.1'),
    'wide448-16x-do': dict(
        implementation='student', config='configs/wide448-dropout.json', run_prefix='wide448-16x-do',
        seeds=[17], batch_size=32, targets=16 * BASELINE_TARGETS,
        note='size x length: width 448 (10.55M) at 16x + dropout 0.1 (asset budget ceiling)'),
    # dropout magnitude at the operating point (batch 2 only had it at 9x)
    'wide256-16x-do15': dict(
        implementation='student', config='configs/wide256-dropout15.json', run_prefix='wide256-16x-do15',
        seeds=[17], batch_size=32, targets=16 * BASELINE_TARGETS,
        note='dropout sweep at 16x: dropout 0.15'),
    'wide256-16x-do20': dict(
        implementation='student', config='configs/wide256-dropout20.json', run_prefix='wide256-16x-do20',
        seeds=[17], batch_size=32, targets=16 * BASELINE_TARGETS,
        note='dropout sweep at 16x: dropout 0.2'),
    # two knobs the project has never varied at all
    'wide256-12x-wd03': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='wide256-12x-wd03',
        seeds=[17], batch_size=32, targets=12 * BASELINE_TARGETS, weight_decay=0.3,
        note='weight decay 0.3 combined with dropout at 12x (wd was confirmed at 3x)'),
    'wide256-12x-warmup500': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='wide256-12x-warmup500',
        seeds=[17], batch_size=32, targets=12 * BASELINE_TARGETS, warmup=500,
        note='warmup sweep: 500 steps instead of 100 (never varied before)'),
    'wide256-16x-batch64': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='wide256-16x-batch64',
        seeds=[17], batch_size=64, targets=16 * BASELINE_TARGETS,
        note='batch-size sweep: 64 instead of 32 at the same 16x target budget'),
    # -----------------------------------------------------------------------
    # Round 7: weight decay 0.3 REOPENED the training-length axis -- at wd 0.1 the
    # curve had saturated (16->24: -0.0015, 24->32: -0.0024) but at wd 0.3 the
    # 16->24 step was still -0.0091.  So the top priority is finding where longer
    # training stops paying.  Training length is free at inference time, and the
    # run at the top of the list is the single most valuable missing data point.
    # -----------------------------------------------------------------------
    'wide256-32x-do-wd03': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='wide256-32x-do-wd03',
        seeds=[17], batch_size=32, targets=32 * BASELINE_TARGETS, weight_decay=0.3,
        note='length axis with wd 0.3: 32x (does the reopened curve keep falling?)'),
    'wide256-48x-do-wd03': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='wide256-48x-do-wd03',
        seeds=[17], batch_size=32, targets=48 * BASELINE_TARGETS, weight_decay=0.3,
        note='length axis with wd 0.3: 48x (brackets the peak from above)'),
    # seeds of the current best recipe, needed for freeze-time statistics and as
    # ensemble/soup members
    'wide256-24x-do-wd03-s18': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='wide256-24x-do-wd03',
        seeds=[18], batch_size=32, targets=24 * BASELINE_TARGETS, weight_decay=0.3,
        note='seed + ensemble member of the best recipe (24x + dropout 0.1 + wd 0.3)'),
    'wide256-24x-do-wd03-s19': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='wide256-24x-do-wd03',
        seeds=[19], batch_size=32, targets=24 * BASELINE_TARGETS, weight_decay=0.3,
        note='seed + ensemble member of the best recipe (24x + dropout 0.1 + wd 0.3)'),
    'wide256-24x-do-wd03-s20': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='wide256-24x-do-wd03',
        seeds=[20], batch_size=32, targets=24 * BASELINE_TARGETS, weight_decay=0.3,
        note='seed + ensemble member of the best recipe (24x + dropout 0.1 + wd 0.3)'),
    # the two best settings combined: dropout 0.15 (better by 3.5 sigma at 16x)
    # on top of weight decay 0.3 (better by ~0.02 at every length)
    'wide256-24x-do15-wd03': dict(
        implementation='student', config='configs/wide256-dropout15.json', run_prefix='wide256-24x-do15-wd03',
        seeds=[17], batch_size=32, targets=24 * BASELINE_TARGETS, weight_decay=0.3,
        note='both best settings combined: dropout 0.15 + weight decay 0.3 at 24x'),
    # -----------------------------------------------------------------------
    # Round 8 (2026-09-24 daytime, GPU kept up all day).  Two facts make this
    # worth a lot of GPU time:
    #   * probability-averaged ensembling gains a LOT here (-0.054 / -0.073 /
    #     -0.084 for 2 / 3 / 4 members) and keeps improving with member count;
    #   * the submitted artefact must be a single checkpoint (the frozen scorer
    #     evaluates exactly one), so the plan is to distil that ensemble.
    # Stronger teacher pool first, then the distillation sweep.
    # -----------------------------------------------------------------------
    'wide256-24x-wd03-more-seeds': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='wide256-24x-do-wd03',
        seeds=[21, 22, 23, 24, 25, 26], batch_size=32, targets=24 * BASELINE_TARGETS,
        weight_decay=0.3,
        note='more ensemble members for the best recipe (the ensemble gain grows with member count)'),
    'wide256-24x-wd03-more-seeds2': dict(
        implementation='student', config='configs/wide256-dropout.json', run_prefix='wide256-24x-do-wd03',
        seeds=[27, 28, 29, 30, 31, 32], batch_size=32, targets=24 * BASELINE_TARGETS,
        weight_decay=0.3,
        note='second batch of ensemble members: lifts the ceiling the distillation can reach'),
}

FIELDS = ['run_id', 'plan', 'method', 'config_path', 'seed', 'parent_checkpoints', 'steps',
          'batch_size', 'context', 'processed_targets_including_ancestry', 'parameters',
          'hardware', 'threads', 'train_precision', 'train_seconds', 'validation_seconds',
          'test_seconds', 'peak_ram_gb', 'peak_gpu_allocated_gb', 'validation_bpb', 'test_bpb',
          'checkpoint_sha256', 'selected_on', 'notes']


def child_peak_rss_gb() -> float:
    """Peak RSS of finished child processes (upper bound; bytes on macOS, KiB on Linux).

    Returns 0.0 where `resource` is unavailable, i.e. on Windows.
    """
    if resource is None:
        return 0.0
    raw = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return raw / (1024 ** 3) if sys.platform == 'darwin' else raw / (1024 ** 2) / 1024


def steps_for(batch_size: int) -> int:
    return BASELINE_TARGETS // (batch_size * CONTEXT)


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def run_dir_for(plan_name: str, seed: int) -> Path:
    return ROOT / 'runs' / f"{PLANS[plan_name]['run_prefix']}-s{seed}"


def cmd_list(_args):
    print(f'baseline target budget: {BASELINE_TARGETS:,} processed targets '
          f'({steps_for(32)} steps x 32 x {CONTEXT})\n')
    for name, spec in PLANS.items():
        steps = spec.get('targets', BASELINE_TARGETS) // (spec['batch_size'] * CONTEXT)
        done = sum((run_dir_for(name, s) / 'metrics.json').exists() for s in spec['seeds'])
        print(f'{name:16s} impl={spec["implementation"]:8s} config={spec["config"]:34s} '
              f'seeds={spec["seeds"]} steps={steps} [{done}/{len(spec["seeds"])} done]')
        print(f'{"":16s} {spec["note"]}')


def guard_config(implementation: str, config_path: Path):
    """Fail fast on a config the model cannot actually run.

    A width that is not a multiple of `heads` builds fine -- the parameter count
    is unchanged -- but blows up on the very first forward pass.  At ~50 minutes
    per run that is an expensive way to learn about an arithmetic slip, so build
    the model and do one tiny forward pass before committing to a run.
    """
    import importlib
    import torch
    config = json.loads(config_path.read_text())
    if config['width'] % config['heads']:
        sys.exit(f'[error] width {config["width"]} is not divisible by heads {config["heads"]}')
    module = importlib.import_module(implementation)
    model = module.build_model(config)
    ids = torch.zeros(2, min(config['context'], 16), dtype=torch.long)
    with torch.no_grad():
        if hasattr(model, 'predict_log_probs'):
            model.predict_log_probs(ids)
        else:
            model(ids)
    print(f'[check] {config_path.name} runs: '
          f'{sum(p.numel() for p in model.parameters()):,} parameters', flush=True)


def cmd_run(args):
    spec = PLANS[args.plan]
    config = ROOT / spec['config']
    if not config.exists():
        sys.exit(f'config not found: {config}')
    if not args.skip_check:
        guard_config(spec['implementation'], config)
    seeds = args.seeds or spec['seeds']
    batch_size = args.batch_size or spec['batch_size']
    budget = spec.get('targets', BASELINE_TARGETS)
    steps = args.steps or budget // (batch_size * CONTEXT)
    targets = steps * batch_size * CONTEXT
    if targets != budget and not args.allow_uneven:
        print(f'WARNING: {targets:,} targets != plan budget {budget:,}; '
              f'pass --allow-uneven to accept.', flush=True)
    if budget != BASELINE_TARGETS:
        print(f'NOTE: this plan trains {budget / BASELINE_TARGETS:.1f}x the baseline budget; '
              f'compare it against the same-budget runs only for the cost side.', flush=True)

    print(f'plan={args.plan} impl={spec["implementation"]} config={spec["config"]} '
          f'seeds={seeds} steps={steps} batch={batch_size} threads={args.threads} '
          f'targets={targets:,}', flush=True)
    for seed in seeds:
        run_id = f"{spec['run_prefix']}-s{seed}"
        run_dir = run_dir_for(args.plan, seed)
        log_path = ROOT / 'logs' / f'{run_id}.log'

        if (run_dir / 'metrics.json').exists() and not args.force:
            print(f'[skip] {run_id}: already finished '
                  f'(validation BPB {(read_json(run_dir / "metrics.json"))["validation"]["bpb"]:.6f})',
                  flush=True)
            continue
        if run_dir.exists() and any(run_dir.iterdir()) and not args.force:
            sys.exit(f'[error] {run_dir} is non-empty but unfinished; re-run with --force '
                     f'to discard it, or move it aside.')
        if args.force and run_dir.exists():
            shutil.rmtree(run_dir)

        command = [PYTHON, 'train.py', '--implementation', spec['implementation'],
                   '--config', spec['config'], '--run-dir', str(run_dir),
                   '--device', args.device, '--threads', str(args.threads),
                   '--seed', str(seed), '--steps', str(steps), '--batch-size', str(batch_size)]
        if args.precision:
            command += ['--precision', args.precision]
        # Optional recipe knobs; only passed when the plan overrides them, so plans
        # without them run the original recipe exactly.
        for key, flag in (('lr', '--lr'), ('weight_decay', '--weight-decay'), ('warmup', '--warmup')):
            if key in spec:
                command += [flag, str(spec[key])]
        print(f'[run ] {run_id}: {" ".join(command[1:])}', flush=True)
        if args.dry_run:
            continue

        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with log_path.open('w') as handle:
            handle.write('$ ' + ' '.join(command) + '\n')
            handle.flush()
            completed = subprocess.run(command, cwd=ROOT, stdout=handle,
                                       stderr=subprocess.STDOUT).returncode
        wall = time.perf_counter() - started

        if completed != 0 or not (run_dir / 'metrics.json').exists():
            sys.exit(f'[fail] {run_id} (exit {completed}); see {log_path}')

        metrics = read_json(run_dir / 'metrics.json')
        (run_dir / 'experiment.json').write_text(json.dumps(
            {**spec, 'plan': args.plan, 'seed': seed, 'steps': steps,
             'batch_size': batch_size, 'threads': args.threads, 'device': args.device,
             'targets': targets, 'wall_seconds': wall,
             'hardware': HARDWARE,
             'peak_ram_gb_children': child_peak_rss_gb()}, indent=2) + '\n')
        print(f'[done] {run_id}: validation BPB {metrics["validation"]["bpb"]:.6f} '
              f'({metrics["train_seconds"]:.0f}s train, {wall:.0f}s wall)', flush=True)

        if args.test:
            test_command = [PYTHON, 'evaluate.py', '--checkpoint', str(run_dir / 'checkpoint.pt'),
                            '--device', 'cpu', '--precision', 'fp32', '--split', 'test']
            with log_path.open('a') as handle:
                handle.write(f'$ {" ".join(test_command)}\n')
                handle.flush()
                subprocess.run(test_command, cwd=ROOT, stdout=handle,
                               stderr=subprocess.STDOUT, check=True)
            print(f'[test] {run_id}: test BPB '
                  f'{read_json(run_dir / "test_cpu_fp32.json")["bpb"]:.6f}', flush=True)

    cmd_log(args)


def run_hardware(run_dir: Path, experiment: dict) -> str:
    """Hardware string for the run log.

    Runs trained on a different machine carry a `remote_device.json` (written by
    hand when the result was imported), and runs trained here carry `hardware`
    in their experiment.json.  Without this the log would mislabel a remote run
    with the local machine's description, which would be a false record.
    """
    override = read_json(run_dir / 'remote_device.json')
    if override:
        return override.get('hardware', HARDWARE)
    return experiment.get('hardware') or HARDWARE


def collect_rows():
    rows = []
    for run_dir in sorted((ROOT / 'runs').glob('*')):
        metrics = read_json(run_dir / 'metrics.json')
        if not metrics:
            continue
        experiment = read_json(run_dir / 'experiment.json') or {}
        test = read_json(run_dir / 'test_cpu_fp32.json')
        validation = metrics['validation']
        rows.append({
            'run_id': run_dir.name,
            'plan': experiment.get('plan', ''),
            'method': metrics['implementation'],
            'config_path': experiment.get('config', ''),
            'seed': metrics['seed'],
            'parent_checkpoints': '',
            'steps': metrics['train_tokens'] // (metrics['config']['context'] * 32),
            'batch_size': 32,
            'context': metrics['config']['context'],
            'processed_targets_including_ancestry': metrics['train_tokens'],
            'parameters': metrics['parameters'],
            'hardware': run_hardware(run_dir, experiment),
            'threads': metrics['threads'],
            'train_precision': metrics['precision'],
            'train_seconds': round(metrics['train_seconds'], 1),
            'validation_seconds': round(validation['seconds'], 1),
            'test_seconds': round(test['seconds'], 1) if test else '',
            'peak_ram_gb': round(experiment.get('peak_ram_gb_children', 0), 2),
            'peak_gpu_allocated_gb': 0,
            'validation_bpb': round(validation['bpb'], 6),
            'test_bpb': round(test['bpb'], 6) if test else '',
            'checkpoint_sha256': metrics['checkpoint_sha256'],
            'selected_on': 'validation',
            'notes': experiment.get('note', ''),
        })
    return rows


def cmd_log(_args):
    rows = collect_rows()
    # Kept outside runs/ because .gitignore drops that directory, and the log
    # is evidence that belongs in the submitted repository.
    out = ROOT / 'run_log.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {out} ({len(rows)} runs)', flush=True)
    if rows:
        print(f'{"run_id":20s} {"val BPB":>9s} {"test BPB":>9s} {"params":>10s} {"train s":>8s}')
        for row in rows:
            print(f'{row["run_id"]:20s} {row["validation_bpb"]:9.6f} '
                  f'{str(row["test_bpb"]):>9s} {row["parameters"]:10d} {row["train_seconds"]:8.0f}')


def cmd_status(_args):
    for name, spec in PLANS.items():
        print(f'\n{name}: {spec["note"]}')
        for seed in spec['seeds']:
            metrics = read_json(run_dir_for(name, seed) / 'metrics.json')
            if metrics:
                print(f'  seed {seed:3d}  done    validation BPB {metrics["validation"]["bpb"]:.6f}')
            else:
                print(f'  seed {seed:3d}  pending')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list').set_defaults(func=cmd_list)
    sub.add_parser('status').set_defaults(func=cmd_status)
    sub.add_parser('log').set_defaults(func=cmd_log)

    run = sub.add_parser('run')
    run.add_argument('plan', choices=list(PLANS))
    run.add_argument('--seeds', type=int, nargs='+')
    run.add_argument('--steps', type=int)
    run.add_argument('--batch-size', type=int)
    run.add_argument('--threads', type=int, default=4)
    run.add_argument('--device', default='cpu',
                     help="pass through to train.py; 'cuda' is much faster if available")
    run.add_argument('--precision', choices=['auto', 'fp32', 'bf16'],
                     help='pass through to train.py; omit to keep its default (auto)')
    run.add_argument('--test', action='store_true',
                     help='also score the test split; use only for a frozen method')
    run.add_argument('--force', action='store_true', help='discard and rerun finished/non-empty dirs')
    run.add_argument('--allow-uneven', action='store_true',
                     help='accept a target count different from the baseline budget')
    run.add_argument('--dry-run', action='store_true')
    run.add_argument('--skip-check', action='store_true',
                     help='skip the pre-flight build/forward check of the config')
    run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
