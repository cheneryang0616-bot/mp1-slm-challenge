"""Print the best (lowest-validation) configuration among the final-length candidates.

Used by the overnight chain to decide which width gets a second seed: the rule is the same as
everywhere else in this project -- **selection on the validation split only**.

Output: one tab-separated line  <name> \\t <config path> \\t <validation BPB>
Diagnostics (which candidates were available) go to stderr so they can be logged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CANDIDATES = {
    'wide384': ('configs/wide384-nodrop.json', 'runs/wide384-T1-58k-wd0-s17/metrics.json'),
    'wide448': ('configs/wide448-nodrop.json', 'runs/wide448-T1-58k-wd0-s17/metrics.json'),
    'wide512': ('configs/wide512-nodrop.json', 'runs/wide512-T1-58k-wd0-s17/metrics.json'),
}

FALLBACK = ('wide384', float('nan'), 'configs/wide384-nodrop.json')


def main() -> None:
    best = None
    for name, (config, metrics) in CANDIDATES.items():
        try:
            value = json.loads(Path(metrics).read_text())['validation']['bpb']
        except Exception as exc:                      # missing / half-written / unreadable
            print(f'{name}: unavailable ({exc})', file=sys.stderr)
            continue
        print(f'{name}: validation BPB {value:.6f}', file=sys.stderr)
        if best is None or value < best[1]:
            best = (name, value, config)
    if best is None:
        print('no candidate metrics readable -> fallback', file=sys.stderr)
        best = FALLBACK
    print(f'{best[0]}\t{best[2]}\t{best[1]}')


if __name__ == '__main__':
    main()
