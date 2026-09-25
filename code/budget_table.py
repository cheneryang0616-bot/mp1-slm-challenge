"""Measure the report's budget table in one session: scoring time + peak RSS + assets.

Why this exists: the three inference budgets must be comparable across models, so every
number in the table must come from the same machine, the same thread count and the same
session (the baseline is included as a control).  `measure_budget.py` cannot do the memory
column correctly -- it reads a *children* high-water mark that drifts upward -- so this
script calls `_peak_wrap.py`, which makes each scoring process report its own ru_maxrss.

Usage (from code/):
    python budget_table.py --threads 4 --split validation --repeat 1
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'runs' / '_shape_probe'

# label -> (checkpoint path, note)
MODELS = {
    'baseline': ('runs/verify-baseline-s17/checkpoint.pt', ''),
    'width 256': ('runs/wide256-6x-s17/checkpoint.pt', ''),
    'width 320': (str(OUT / 'wide320-nodrop.pt'), 'shape probe'),
    'width 384': ('../_server_results/checkpoints/wide384-T1-58k-wd0-s17.pt', '') ,
    'width 448': (str(OUT / 'wide448-nodrop.pt'), 'shape probe'),
    'width 512': ('../_server_results/checkpoints/wide512-T1-58k-wd0-s18.pt', 'submitted'),
    'width 576': (str(OUT / 'wide576-nodrop.pt'), 'shape probe (over budget)'),
}


def run(label: str, checkpoint: Path, split: str, threads: int) -> dict:
    result_path = Path(f'/tmp/budget_{label.replace(" ", "")}_{threads}t.json')
    command = [sys.executable, str(ROOT / '_peak_wrap.py'),
               '--checkpoint', str(checkpoint), '--device', 'cpu', '--precision', 'fp32',
               '--threads', str(threads), '--split', split, '--output', str(result_path)]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if process.returncode != 0:
        return {'label': label, 'error': process.stderr[-300:]}
    payload = json.loads(result_path.read_text())
    peak = re.search(r'PEAK_RSS_MIB\s+([\d.]+)', process.stdout)
    info = json.loads((Path(checkpoint.parent) / 'metrics.json').read_text()) if False else None
    del info
    return {
        'label': label,
        'assets_mib': checkpoint.stat().st_size / 2**20,
        'seconds': payload['seconds'],
        'bpb': payload['bpb'],
        'peak_mib': float(peak.group(1)) if peak else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='validation')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()

    rows = []
    for label, (path, note) in MODELS.items():
        checkpoint = (ROOT / path).resolve()
        if not checkpoint.is_file():
            print(f'# 跳过 {label}: 无 checkpoint {checkpoint}', file=sys.stderr)
            continue
        for _ in range(args.repeat):
            row = run(label, checkpoint, args.split, args.threads)
            if 'error' in row:
                print(f'# {label} 失败: {row["error"]}', file=sys.stderr)
                continue
            row['note'] = note
            rows.append(row)
            print(f'# {label}: {row["seconds"]:.2f} s, {row["peak_mib"]:.0f} MiB peak',
                  file=sys.stderr)

    baseline = next((r for r in rows if r['label'] == 'baseline'), None)
    if baseline is None:
        raise SystemExit('no baseline measurement')
    print()
    print(f'| Model | Parameters | Assets | Scoring time | x baseline | Peak RSS |')
    print(f'|---|---:|---:|---:|---:|---:|')
    for row in rows:
        ratio = row['seconds'] / baseline['seconds']
        params = {'baseline': 1088256, 'width 256': 3673344, 'width 320': 5572160,
                  'width 384': 7867776, 'width 448': 10557120, 'width 512': 13634048,
                  'width 576': 17110080}.get(row['label'], 0)
        print(f"| {row['label']} | {params:,} | {row['assets_mib']:.2f} MiB | "
              f"{row['seconds']:.2f} s | {ratio:.2f} | {row['peak_mib']/1024:.2f} GiB |")
    print()
    print(f"baseline = {baseline['seconds']:.2f} s, limit {5*baseline['seconds']:.1f} s "
          f"({args.threads} threads, {args.split} split)")
    (ROOT / 'runs' / 'budget_table.json').write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
