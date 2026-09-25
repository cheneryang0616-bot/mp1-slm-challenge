"""Log-probability ensemble scorer for MP1.

Why this exists
---------------
The official scorer `evaluate.py` is frozen by the assignment and scores exactly
one checkpoint, so it cannot express an ensemble.  This script re-implements the
*exact same* measurement (same `common.load_data`, same `common.windows`, same
byte accounting, same FP32 CPU path) but takes several checkpoints of the same
architecture and averages their per-token log-probabilities before computing the
negative log-likelihood.

How the members are combined matters and is easy to get wrong:

* Averaging *probabilities*,  p_ens(y) = (1/M) sum_m p_m(y),  is the standard
deep-ensemble rule.  By Jensen's inequality its NLL is at most the mean member
NLL, so it can actually improve on every member.
* Averaging *log-probabilities* (a geometric mean) looks similar but is a no-op:
the resulting NLL is *exactly* the arithmetic mean of the members' NLLs, so it
never beats the average member.  It is reported here only as a diagnostic.

Both combined distributions are normalised, so the BPB is directly comparable
to the single-model number from `evaluate.py`.

This file is an *addition*, not a modification: `evaluate.py` and `common.py`
are untouched.

Usage
-----
    python ensemble_eval.py --split validation \
        --checkpoints runs/wide256-s17/checkpoint.pt runs/wide256-s18/checkpoint.pt \
        --output runs/ensemble-2-val.json

    # also report each member's own BPB (sanity check that members are similar)
    python ensemble_eval.py --split validation --per-member --checkpoints a.pt b.pt
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from common import (PROTOCOL, ROOT, autocast, device_metrics, load_data,
                    make_model, setup, sha, windows)


def load_member(path: Path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['protocol'] != PROTOCOL:
        raise ValueError(f'{path} belongs to a different course protocol.')
    model, implementation_sha = make_model(checkpoint['implementation'],
                                           checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'])
    return model, checkpoint, implementation_sha


@torch.no_grad()
def score_members(models, tokens, byte_count, device, precision, batch_size=32):
    """One forward pass per member over the split.

    Returns the probability-averaged ensemble BPB, each member's own BPB, and
    the (no-op) log-prob-averaged BPB as a diagnostic.
    """
    for model in models:
        model.eval()
    started = time.perf_counter()
    member_count = len(models)
    nll = 0.0                                    # probability-averaged ensemble
    logmean_nll = 0.0                            # log-prob average (diagnostic)
    member_nll = [0.0] * member_count
    count = 0
    for x, y in windows(tokens, batch_size):
        x, y = x.to(device), y.to(device)
        logps = []
        for model in models:
            with autocast(device, precision):
                logp = model.predict_log_probs(x).float()
            if logp.shape != (*x.shape, 2048) or not torch.isfinite(logp).all():
                raise ValueError('Model returned non-finite log-probabilities of the wrong shape.')
            if torch.logsumexp(logp, dim=-1).abs().max().item() > 1e-3:
                raise ValueError('The output is not a normalized probability distribution.')
            logps.append(logp)
        target = y.clamp_min(0).unsqueeze(-1)
        mask = (y != -100)
        for index, logp in enumerate(logps):
            losses = -logp.gather(-1, target).squeeze(-1)
            member_nll[index] += losses.masked_fill(~mask, 0).double().sum().item()

        stacked = torch.stack(logps)                    # [M, batch, time, vocab]
        # p_ens = mean_m p_m  ->  log p_ens = logsumexp_m(log p_m) - log M
        combined = torch.logsumexp(stacked, dim=0) - math.log(member_count)
        losses = -combined.gather(-1, target).squeeze(-1)
        nll += losses.masked_fill(~mask, 0).double().sum().item()

        geometric = stacked.mean(dim=0)                 # mean of log-probs
        losses = -geometric.gather(-1, target).squeeze(-1)
        logmean_nll += losses.masked_fill(~mask, 0).double().sum().item()
        count += mask.sum().item()
    seconds = time.perf_counter() - started
    to_bpb = lambda total: total / math.log(2) / byte_count  # noqa: E731
    return {
        'bpb': to_bpb(nll),
        'members_bpb': [to_bpb(value) for value in member_nll],
        'mean_member_bpb': to_bpb(sum(member_nll) / member_count),
        'logprob_average_bpb': to_bpb(logmean_nll),
        'targets': count,
        'utf8_bytes': byte_count,
        'seconds': seconds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoints', nargs='+', required=True, type=Path)
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--precision', choices=['auto', 'fp32', 'bf16'], default='fp32')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--per-member', action='store_true',
                        help='print each member checkpoint BPB alongside the ensemble')
    args = parser.parse_args()

    device, precision = setup(args.device, args.precision, args.threads)
    models, configs, shas = [], [], []
    for path in args.checkpoints:
        model, checkpoint, implementation_sha = load_member(path, device)
        models.append(model)
        configs.append(checkpoint['config'])
        shas.append(sha(path))
    # Members may have different shapes (a heterogeneous ensemble is allowed and
    # adds diversity); we only require that each checkpoint was trained for the
    # same protocol and the same context window, which score_members checks.
    if len({config['context'] for config in configs}) != 1:
        raise ValueError('All ensemble members must use the same context length.')
    if len({json.dumps(config, sort_keys=True) for config in configs}) > 1:
        print(f'note: heterogeneous ensemble over {len({json.dumps(c, sort_keys=True) for c in configs})} '
              f'distinct configs', flush=True)

    data = load_data()
    result = score_members(models, *data[args.split], device, precision)
    result.update(
        protocol=PROTOCOL,
        split=args.split,
        precision=precision,
        members=[str(path) for path in args.checkpoints],
        member_sha256=shas,
        member_configs=configs,
        member_parameters=[sum(p.numel() for p in model.parameters()) for model in models],
        combined_parameters=sum(p.numel() for model in models for p in model.parameters()),
        checkpoint_bytes=sum(path.stat().st_size for path in args.checkpoints),
        evaluator_sha256=sha(Path(__file__)),
        tokenizer_sha256=sha(ROOT / 'data/tokenizer.json'),
        **device_metrics(device),
    )
    output = args.output or (ROOT / 'runs' / f'ensemble-{len(models)}-{args.split}.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not args.per_member:
        result.pop('members_bpb', None)
    print(f'\nensemble ({len(models)} members) {args.split} BPB = {result["bpb"]:.6f} '
          f'in {result["seconds"]:.1f}s -> {output}')


if __name__ == '__main__':
    main()
