"""Checkpoint averaging (a "model soup") for MP1.

Averaging the weights of several fine-tuned models that share an architecture
often lands in a flatter region of the loss surface and generalises a little
better than any single member.  Unlike `ensemble_eval.py` this produces a
*single* checkpoint, so the frozen official scorer `evaluate.py` can score it
directly -- no custom scorer is needed at submission time.

The members must come from the same architecture and the same training recipe
(here: the same config, different seeds), otherwise the average is meaningless.

Usage
-----
    python soup.py --output runs/wide256-soup/checkpoint.pt \
        --checkpoints runs/wide256-s17/checkpoint.pt runs/wide256-s18/checkpoint.pt \
                      runs/wide256-s19/checkpoint.pt

    # then, with the untouched official scorer:
    python evaluate.py --checkpoint runs/wide256-soup/checkpoint.pt \
        --device cpu --precision fp32 --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common import PROTOCOL, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoints', nargs='+', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--weights', nargs='+', type=float,
                        help='optional per-member weights (default: uniform)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    checkpoints = [torch.load(path, map_location='cpu', weights_only=True) for path in args.checkpoints]
    for path, checkpoint in zip(args.checkpoints, checkpoints):
        if checkpoint['protocol'] != PROTOCOL:
            raise ValueError(f'{path} belongs to a different course protocol.')
    configs = [checkpoint['config'] for checkpoint in checkpoints]
    # Only the *parameter layout* has to match: dropout (and other non-parameter
    # settings) may legitimately differ between soup members, because they do not
    # change any tensor shape.  The state-dict key/shape check below is the real
    # guarantee, so this is a cheap early sanity check on the architecture keys.
    arch_keys = ('vocab', 'context', 'width', 'heads', 'depth', 'norm', 'mlp', 'pos', 'bias')
    archs = [{key: config.get(key) for key in arch_keys} for config in configs]
    if any(arch != archs[0] for arch in archs):
        raise ValueError('All soup members must share the same architecture config.')
    if len({json.dumps(config, sort_keys=True) for config in configs}) > 1:
        print('note: members differ in non-parameter settings (e.g. dropout); '
              'parameter shapes match, so the average is still well defined', flush=True)

    weights = args.weights or [1.0] * len(checkpoints)
    if len(weights) != len(checkpoints):
        raise ValueError('Provide one weight per checkpoint.')
    total = sum(weights)
    weights = [weight / total for weight in weights]

    keys = list(checkpoints[0]['model'].keys())
    for index, checkpoint in enumerate(checkpoints[1:], start=1):
        if list(checkpoint['model'].keys()) != keys:
            raise ValueError(f'State-dict keys differ between member 0 and member {index}.')

    averaged = {}
    for key in keys:
        reference = checkpoints[0]['model'][key]
        if not reference.is_floating_point():
            averaged[key] = reference.clone()
            continue
        acc = torch.zeros_like(reference, dtype=torch.float64)
        for weight, checkpoint in zip(weights, checkpoints):
            acc += weight * checkpoint['model'][key].to(torch.float64)
        averaged[key] = acc.to(reference.dtype)

    # Report the per-tensor drift so we can see the soup is not degenerate.
    drift = max((averaged[key] - checkpoints[0]['model'][key]).abs().max().item()
                for key in keys if averaged[key].is_floating_point())
    print(f'members: {len(checkpoints)}  max |soup - member0| = {drift:.5f}')
    if args.dry_run:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {**checkpoints[0], 'model': averaged,
               'soup_members': [str(path) for path in args.checkpoints],
               'soup_weights': weights,
               'soup_member_sha256': [sha(path) for path in args.checkpoints]}
    torch.save(payload, args.output)
    print(f'wrote {args.output} ({args.output.stat().st_size / 2**20:.2f} MiB) '
          f'sha256 {sha(args.output)}')
    print(json.dumps({key: payload[key] for key in ('protocol', 'implementation', 'config',
                                                    'soup_weights')}, indent=2))


if __name__ == '__main__':
    main()
