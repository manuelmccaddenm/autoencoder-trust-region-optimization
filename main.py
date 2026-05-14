from lib_benchmark import benchmark, print_table


def main():
    rows = benchmark(ds=(3, 5, 10, 100), lbfgs_m=10, maxIter=500)
    print_table(rows)


if __name__ == "__main__":
    main()
