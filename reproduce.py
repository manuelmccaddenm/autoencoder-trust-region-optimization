"""
Reproduce todos los resultados del reporte.

Uso:
  python reproduce.py            # todo (~6 min)
  python reproduce.py fast       # sin stress test (~10 s)
"""

import subprocess
import sys
import time


def run(label, script, *args):
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    t0 = time.perf_counter()
    subprocess.run([sys.executable, script, *args], check=True)
    print(f"  -> {label}: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    fast = "fast" in sys.argv[1:]
    run("1. Benchmark", "main.py")
    run("2. Figuras", "make_figs.py")
    if not fast:
        run("3. Stress test (5000 corridas)", "stress_test.py", "5000")
    print("\nListo.")
