# -*- coding: utf-8 -*-
"""
main.py
Benchmark del autoencoder lineal d -> 2 -> d con TR + Dogleg, comparando
BFGS (matriz completa) vs L-BFGS (m=10) sobre d in {3, 5, 10, 100}.
"""

from lib_benchmark import benchmark, print_table


def main():
    rows = benchmark(ds=(3, 5, 10, 100), lbfgs_m=10, maxIter=500)
    print_table(rows)


if __name__ == "__main__":
    main()
