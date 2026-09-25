"""冻结选型：只用 validation 选 checkpoint，再对官方 scorer 评一次 test。

规则依据（GUIDE.md）：
  * 选型只能用 validation；test 只在配置冻结后评一次。
  * 评测必须走未修改的 `evaluate.py`（FP32 / CPU）。
  * 三条推理预算：CPU 评分时间 <= 5x 基线、峰值内存 <= 4 GiB、推理资产 <= 64 MiB。

这个脚本把"最后一步"变成一条命令：
  1. 读每个候选 run 的 `metrics.json`，按 **validation BPB** 排序、选最小者；
  2. 用官方 `evaluate.py --device cpu --precision fp32 --split test` 评它（子进程调用，
     保证走的是原版评分路径）；
  3. 用同一台机器、同一线程数再评一次 `configs/baseline.json` 的 checkpoint，
     得到时间比（预算口径）；
  4. 打印 report 用的 Markdown 表 + 写 JSON。

用法（checkpoint 拉回本地之后）：
    .venv/bin/python select_final.py \
        --metrics-root ../../_server_results/runs \
        --ckpt-root ../../_server_results/checkpoints \
        wide384-T1-58k-wd0-s17 wide384-T1-58k-wd0-s18 wide384-T1-58k-wd0-s19

默认 `--split test` 只评被选中的那一个；`--also-test-all` 会把所有候选的 test 分数
也算出来放进附录（用于 seed 方差展示；选型仍然只用 validation，这一点会在输出里注明）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASELINE_CKPT = ROOT / 'runs' / 'verify-baseline-s17' / 'checkpoint.pt'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def find_checkpoint(name: str, ckpt_root: Path | None, metrics_root: Path) -> Path | None:
    for candidate in ([ckpt_root / f'{name}.pt'] if ckpt_root else []) + [
            metrics_root / name / 'checkpoint.pt']:
        if candidate.is_file():
            return candidate
    return None


def run_official(checkpoint: Path, split: str, threads: int, precision: str) -> dict:
    output = checkpoint.parent / f'{split}_{checkpoint.stem}_official.json'
    command = [sys.executable, 'evaluate.py', '--checkpoint', str(checkpoint),
               '--device', 'cpu', '--precision', precision, '--threads', str(threads),
               '--split', split, '--output', str(output)]
    started = time.perf_counter()
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    wall = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(f'evaluate.py failed for {checkpoint}:\n{process.stderr[-2000:]}')
    result = json.loads(output.read_text())
    result['wall_seconds'] = wall
    result['command'] = ' '.join(command)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('candidates', nargs='+', help='run 目录名（如 wide384-T1-58k-wd0-s17）')
    parser.add_argument('--metrics-root', type=Path, default=ROOT / 'runs')
    parser.add_argument('--ckpt-root', type=Path, default=None,
                        help='checkpoint 所在目录（<name>.pt）；不给则在 run 目录里找')
    parser.add_argument('--baseline-checkpoint', type=Path, default=BASELINE_CKPT)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--precision', default='fp32')
    parser.add_argument('--split', default='test')
    parser.add_argument('--also-test-all', action='store_true')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs' / 'final_selection.json')
    args = parser.parse_args()

    rows = []
    for name in args.candidates:
        metrics_path = args.metrics_root / name / 'metrics.json'
        if not metrics_path.is_file():
            print(f'!! 跳过 {name}：缺少 {metrics_path}')
            continue
        metrics = json.loads(metrics_path.read_text())
        checkpoint = find_checkpoint(name, args.ckpt_root, args.metrics_root)
        rows.append({
            'run': name,
            'validation_bpb': metrics['validation']['bpb'],
            'parameters': metrics.get('parameters'),
            'seed': metrics.get('seed'),
            'steps': metrics.get('train_tokens', 0) // (256 * 32),
            'checkpoint': str(checkpoint) if checkpoint else None,
            'checkpoint_mib': checkpoint.stat().st_size / 2**20 if checkpoint else None,
            'checkpoint_sha256': sha256(checkpoint) if checkpoint else None,
        })

    if not rows:
        raise SystemExit('没有可用候选')

    print('\n## 候选（按 validation BPB 排序；选型只用这一列）\n')
    print('| run | seed | steps | params | validation BPB | 资产 MiB |')
    print('|---|---:|---:|---:|---:|---:|')
    for row in sorted(rows, key=lambda r: r['validation_bpb']):
        print('| {} | {} | {} | {} | {:.6f} | {} |'.format(
            row['run'], row['seed'], row['steps'], f"{row['parameters']:,}",
            row['validation_bpb'], f"{row['checkpoint_mib']:.2f}" if row['checkpoint_mib'] else 'n/a'))

    values = [r['validation_bpb'] for r in rows]
    if len(values) > 1:
        print(f'\nseed 之间：均值 {statistics.mean(values):.6f}，'
              f'样本 σ {statistics.stdev(values):.6f}，极差 {max(values) - min(values):.6f}')

    frozen = min(rows, key=lambda r: r['validation_bpb'])
    print(f"\n冻结/提交候选（validation 最小）：**{frozen['run']}** "
          f"val BPB {frozen['validation_bpb']:.6f}")
    if not frozen['checkpoint']:
        raise SystemExit(f"找不到 {frozen['run']} 的 checkpoint；先把 checkpoint 拉回本地")

    print(f"\n--- 官方评测（{args.split} / cpu / {args.precision} / {args.threads} threads）---")
    baseline = run_official(args.baseline_checkpoint, args.split, args.threads, args.precision)
    selected = run_official(Path(frozen['checkpoint']), args.split, args.threads, args.precision)
    ratio = selected['seconds'] / baseline['seconds']

    print('| 模型 | {} BPB | 评分秒 | x基线 | 资产 MiB |'.format(args.split))
    print('|---|---:|---:|---:|---:|')
    print(f"| baseline | {baseline['bpb']:.6f} | {baseline['seconds']:.2f} | 1.00 | "
          f"{args.baseline_checkpoint.stat().st_size / 2**20:.2f} |")
    print(f"| {frozen['run']} | {selected['bpb']:.6f} | {selected['seconds']:.2f} | "
          f"{ratio:.2f} | {frozen['checkpoint_mib']:.2f} |")

    appendix = None
    if args.also_test_all:
        appendix = []
        for row in rows:
            if not row['checkpoint'] or row['run'] == frozen['run']:
                continue
            result = run_official(Path(row['checkpoint']), args.split, args.threads, args.precision)
            appendix.append({'run': row['run'], 'validation_bpb': row['validation_bpb'],
                             'test_bpb': result['bpb'], 'seconds': result['seconds']})
            print(f"[附录] {row['run']}: val {row['validation_bpb']:.6f} -> "
                  f"{args.split} {result['bpb']:.6f}")

    payload = {
        'protocol': '7506-mp1-wt2-v2',
        'selected_on': 'validation',
        'frozen_run': frozen['run'],
        'frozen_validation_bpb': frozen['validation_bpb'],
        'test_split': args.split,
        'test_bpb': selected['bpb'],
        'test_seconds': selected['seconds'],
        'baseline_bpb': baseline['bpb'],
        'baseline_seconds': baseline['seconds'],
        'cost_ratio_vs_baseline': ratio,
        'asset_mib': frozen['checkpoint_mib'],
        'checkpoint_sha256': frozen['checkpoint_sha256'],
        'candidates': rows,
        'test_appendix_other_seeds': appendix,
        'note': ('checkpoint 由 validation 选出；test 只作为冻结后的最终测量。'
                 '附录里的其它 seed 分数不参与选型。'),
        'measured_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f'\n已写入 {args.out}')


if __name__ == '__main__':
    main()
