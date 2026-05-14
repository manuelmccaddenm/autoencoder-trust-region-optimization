"""
Reproduces all results from the report.

Usage:
  python reproduce.py            # everything (~6 min)
  python reproduce.py fast       # skip stress test (~10 s)
"""

import subprocess
import sys
import time


def run(label, script):
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    t0 = time.perf_counter()
    subprocess.run([sys.executable, script], check=True)
    print(f"  -> {label}: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    fast = "fast" in sys.argv[1:]
    run("1. Benchmark", "main.py")
    run("2. Figures", "make_figs.py")
    if not fast:
        run("3. Stress test (5000 runs)", "stress_test.py")
    print("\nDone.")
