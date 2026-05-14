"""
Reproduce todos los resultados del reporte.

Uso:
  python reproduce.py            # todo (~6 min)
  python reproduce.py fast       # sin stress test (~10 s)
"""

import subprocess
import sys
import time


def correr(etiqueta, script, *args):
    print(f"\n{'='*60}\n  {etiqueta}\n{'='*60}")
    t0 = time.perf_counter()
    subprocess.run([sys.executable, script, *args], check=True)
    print(f"  -> {etiqueta}: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    rapido = "fast" in sys.argv[1:]
    correr("1. Benchmark", "main.py")
    correr("2. Figuras", "make_figs.py")
    if not rapido:
        correr("3. Stress test (5000 corridas)", "stress_test.py", "5000")
    print("\nListo.")
