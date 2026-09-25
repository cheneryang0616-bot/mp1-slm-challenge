"""Measure the three inference budgets for a set of checkpoints.

The assignment caps, for the scorer alone:

    uncompressed inference assets  <= 64 MiB
    CPU scoring time               <= 5x the classroom baseline
    peak evaluation RAM            <= 4 GiB

This script reports all three in the same units the REPORT.md budget table uses,
so the numbers drop straight into the report.

Why the validation split, not test
----------------------------------
The test split may only be scored once, for the frozen final model (and only
after every method choice is locked).  The scoring-time budget is a *ratio*
against the classroom baseline measured on the same machine with the same thread
count, and that ratio is essentially split-independent (both splits are ~1.1 MB
of contiguous WikiText-2), so we measure on validation and say so.

Usage
-----
    python measure_budget.py --threads 4 \
        --checkpoint baseline:runs/baseline-s17/checkpoint.pt \
        --checkpoint w256:runs/wide256-T1-58k-wd0-nd-s17/checkpoint.pt
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MIB = 1024 ** 2


def measure(label: str, checkpoint: Path, split: str, threads: int, device: str,
            precision: str, out_json: Path) -> dict:
    command = [
        sys.executable, str(ROOT / 'evaluate.py'),
        '--checkpoint', str(checkpoint),
        '--split', split,
        '--threads', str(threads),
        '--device', device,
        '--precision', precision,
        '--output', str(out_json),
    ]
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    # ru_maxrss is a high-water mark, so `after` is the peak of *any* child so
    # far -- report it as an upper bound and note that it only grows.
    payload = json.loads(out_json.read_text())
    return {
        'label': label,
        'checkpoint': str(checkpoint),
        'assets_mib': checkpoint.stat().st_size / MIB,
        'seconds': payload.get('seconds'),
        'bpb': payload.get('bpb'),
        'child_peak_rss_gib_upper_bound': max(after, before) / (1024 ** 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='measure the MP1 inference budgets')
    parser.add_argument('--checkpoint', action='append', default=[],
                        help='label:path (repeatable); the first one is treated as the baseline')
    parser.add_argument('--split', default='validation')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--precision', default='fp32')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    rows = []
    for item in args.checkpoint:
        label, _, path = item.partition(':')
        rows.append(measure(label, Path(path), args.split, args.threads,
                            args.device, args.precision,
                            Path(args.output or '/tmp/budget.json')))

    baseline = rows[0]
    print()
    print(f'{"model":<34s} {"assets(MiB)":>11s} {"score(s)":>9s} {"x baseline":>11s} {"peak RSS(GiB)":>13s}  bpb')
    for row in rows:
        ratio = row['seconds'] / baseline['seconds']
        print(f"{row['label']:<34s} {row['assets_mib']:>11.2f} {row['seconds']:>9.2f} "
              f"{ratio:>11.2f} {row['child_peak_rss_gib_upper_bound']:>13.2f}  {row['bpb']:.6f}")
    print()
    print(f'budget limits: assets <= 64 MiB, score time <= 5x baseline '
          f'({5 * baseline["seconds"]:.1f} s here), peak RAM <= 4 GiB')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
