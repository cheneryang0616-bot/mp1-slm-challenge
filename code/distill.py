"""Distil an ensemble of checkpoints into ONE checkpoint.

Why this exists
---------------
Probability-averaging M of our checkpoints buys a large validation gain on this
task.  Measured on the width-256 @ 16x recipe (4 different seeds):

    2 members  1.483437   (-0.0539 against the member mean)
    3 members  1.465464   (-0.0734)
    4 members  1.455747   (-0.0841)

Two constraints stop us from simply submitting the ensemble:

1. Inference budget.  The assignment caps CPU scoring at 5x the classroom
   baseline.  Measured on one machine with one thread count, a single width-256
   model already scores at 1.98x the baseline, so two members cost 3.96x (inside
   the cap) but three cost 5.9x (outside it).
2. Reproducibility of the submission.  The frozen scorer `evaluate.py` scores
   exactly one checkpoint, so an ensemble number could not be reproduced from the
   submitted artefact -- and an unreproducible number is indistinguishable from a
   mis-reported one.

Distillation removes both: one student is trained to match the ensemble's
predictive distribution, so the submitted model is an ordinary single checkpoint
that the untouched official scorer evaluates, at 1x the inference cost.

Nothing here bends the assignment rules: same data split, same tokenizer, same
architecture, training time unrestricted.  All teachers are our own models,
trained from scratch on the provided training split.

Objective
---------
For a batch of windows, teacher probs are the arithmetic mean
    p_T = (1/M) * sum_m softmax(logits_m)
(probability averaging, the same rule as `ensemble_eval.py`), and the student is
fitted to them with the cross-entropy
    L = -sum_v p_T(v) * log p_S(v)
which equals KL(p_T || p_S) up to a constant that does not depend on the student.
Hard-label cross-entropy can be mixed in with `--hard-weight` (default 0).

Usage
-----
    python distill.py --run-dir runs/wide256-distilled-s17 \\
        --config configs/wide256-dropout.json \\
        --teachers runs/wide256-24x-do-wd03-s17/checkpoint.pt \\
                   runs/wide256-24x-do-wd03-s18/checkpoint.pt ... \\
        --steps 14400 --seed 17 --device cuda --precision fp32 --threads 8
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from common import (PROTOCOL, ROOT, autocast, device_metrics, load_data,
                    make_model, setup, sha)
from evaluate import score


def load_teacher(path: Path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['protocol'] != PROTOCOL:
        raise ValueError(f'{path} belongs to a different course protocol.')
    model, _ = make_model(checkpoint['implementation'], checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint['config']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/wide256-dropout.json')
    parser.add_argument('--teachers', nargs='+', required=True, type=Path)
    parser.add_argument('--implementation', default='student')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--precision', choices=['auto', 'fp32', 'bf16'], default='fp32')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--steps', type=int, default=14400)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.3)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--hard-weight', type=float, default=0.0,
                        help='mix in this much hard-label cross-entropy (0 = pure distillation)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='distillation temperature. T=1 is nearly hard-label training on a '
                             '2048-way softmax (the top token dominates), which transfers almost no '
                             'ensemble knowledge; T=2..8 softens the target and exposes the '
                             '"dark knowledge" in the tail.')
    parser.add_argument('--note', default='ensemble distillation')
    args = parser.parse_args()

    total_started = time.perf_counter()
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)

    prepared = time.perf_counter()
    data = load_data()
    tokens = data['train'][0].to(device)
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    teachers, teacher_configs = [], []
    for path in args.teachers:
        teacher, teacher_config = load_teacher(path, device)
        # Teachers only have to agree on the vocabulary (the distribution they
        # predict over) and the context length (the windows they see).  They may
        # be smaller or larger than the student -- a *larger* student is in fact
        # the interesting case, since a bigger student has more capacity to
        # represent the ensemble's averaged distribution.
        if teacher_config['vocab'] != config['vocab'] or teacher_config['context'] != config['context']:
            raise ValueError(f'teacher {path} predicts over a different vocabulary/context')
        teachers.append(teacher)
        teacher_configs.append(teacher_config)
    preparation_seconds = time.perf_counter() - prepared
    print(json.dumps({'teachers': len(teachers), 'student_steps': args.steps,
                      'preparation_seconds': preparation_seconds}), flush=True)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = torch.Generator().manual_seed(args.seed)
    log_every = max(1, args.steps // 36)

    started = time.perf_counter()
    history = []
    model.train()
    for step in range(args.steps):
        starts = torch.randint(len(tokens) - 257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:, None] + torch.arange(257, device=device)]
        inputs, targets = batch[:, :-1], batch[:, 1:]

        with torch.no_grad():
            teacher_probs = None
            for teacher in teachers:
                with autocast(device, precision):
                    logits = teacher(inputs).float()
                probs = torch.softmax(logits / args.temperature, dim=-1)
                teacher_probs = probs if teacher_probs is None else teacher_probs + probs
            teacher_probs = (teacher_probs / len(teachers)).detach()

        learning_rate = args.lr * min(1., (step + 1) / args.warmup) * \
            (.1 + .9 * .5 * (1 + math.cos(math.pi * step / args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate

        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            logits = model(inputs).float()
        log_probs = torch.log_softmax(logits / args.temperature, dim=-1)
        # Hinton's T^2 factor: it keeps the gradient magnitude of the soft term
        # on the same scale as the T=1 loss, so one lr works for every T.
        loss = -(teacher_probs * log_probs).sum(-1).mean() * (args.temperature ** 2)
        if args.hard_weight:
            loss = loss + args.hard_weight * F.cross_entropy(
                logits.flatten(0, 1), targets.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

        if (step + 1) % log_every == 0 or step + 1 == args.steps:
            row = {'step': step + 1, 'loss': loss.item(),
                   'seconds': time.perf_counter() - started}
            history.append(row)
            print(json.dumps(row), flush=True)

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started

    validation = score(model, *data['validation'], device, 'fp32')
    validation.pop('window_nll_nats')
    checkpoint = args.run_dir / 'checkpoint.pt'
    torch.save({'protocol': PROTOCOL, 'implementation': args.implementation, 'config': config,
                'model': model.cpu().state_dict(), 'seed': args.seed,
                'train_tokens': args.steps * args.batch_size * config['context'],
                'distilled_from': [str(path) for path in args.teachers]}, checkpoint)
    result = {'protocol': PROTOCOL, 'implementation': args.implementation, 'config': config,
              'seed': args.seed, 'lr': args.lr, 'weight_decay': args.weight_decay,
              'warmup': args.warmup, 'hard_weight': args.hard_weight,
              'temperature': args.temperature,
              'parameters': sum(p.numel() for p in model.parameters()), 'precision': precision,
              'train_tokens': args.steps * args.batch_size * config['context'],
              'preparation_seconds': preparation_seconds, 'train_seconds': train_seconds,
              'validation': validation, 'history': history,
              'distilled_from': [str(path) for path in args.teachers],
              'process_seconds': time.perf_counter() - total_started,
              'torch_version': str(torch.__version__), 'threads': args.threads,
              'checkpoint_sha256': sha(checkpoint), 'implementation_sha256': implementation_sha,
              **device_metrics(device)}
    (args.run_dir / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.run_dir / 'experiment.json').write_text(json.dumps({
        'plan': 'distill', 'seed': args.seed, 'steps': args.steps,
        'batch_size': args.batch_size, 'threads': args.threads, 'device': args.device,
        'temperature': args.temperature, 'hard_weight': args.hard_weight,
        'targets': args.steps * args.batch_size * config['context'],
        'wall_seconds': time.perf_counter() - total_started,
        'hardware': f'{device.type} distillation',
        'note': args.note}, indent=2) + '\n')
    print(json.dumps(result | {'history': []}, indent=2), flush=True)


if __name__ == '__main__':
    main()
