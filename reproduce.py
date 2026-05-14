# -*- coding: utf-8 -*-
"""
reproduce.py
Reproduce todos los resultados del reporte en una sola corrida:

  1. Benchmark iBFGS vs L-BFGS (Tabla 1)         -- ~4 s
  2. Figuras 1-4                                  -- ~3 s
  3. Stress test, 5000 configuraciones aleatorias -- ~6 min

Uso:
  python reproduce.py            # corre todo
  python reproduce.py fast       # omite stress test
"""

import subprocess
import sys
import time


def run(label, cmd):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    subprocess.run([sys.executable, cmd], check=True)
    print(f"  -> {label}: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    fast = "fast" in sys.argv[1:]
    run("1. Benchmark (Tabla 1)", "main.py")
    run("2. Figuras del reporte", "make_figs.py")
    if not fast:
        run("3. Stress test (5000 corridas)", "stress_test.py")
    print("\nListo.")
