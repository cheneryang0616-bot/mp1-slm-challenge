"""Measure *one* model's peak RSS by making the child report its own high-water mark.

`measure_budget.py` reads `RUSAGE_CHILDREN.ru_maxrss`, which is a high-water mark over
*all* children of the shell so far -- so its memory column drifts upward and cannot
distinguish two models of similar size.  This wrapper runs `evaluate.py` in-process via
runpy and prints the peak RSS of that single process, so each model gets its own number.

Usage:
    python _peak_wrap.py --checkpoint <ckpt> --device cpu --precision fp32 --threads 4 --split validation
(the arguments are passed straight through to evaluate.py)
"""
import resource
import runpy
import sys
from pathlib import Path

if __name__ == '__main__':
    sys.argv[0] = str(Path(__file__).resolve().parent / 'evaluate.py')
    try:
        runpy.run_path(sys.argv[0], run_name='__main__')
    finally:
        print(f'PEAK_RSS_MIB {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f}')
