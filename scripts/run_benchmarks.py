"""Benchmark the discrete ABC (or the bit-flip baseline) on all instances.

Usage:
    python scripts/run_benchmarks.py [--runs 10] [--method abc|bitflip] [--large]
"""

import argparse

from bee_knapsack import P08, PISINGER, load_pisinger, run_repeated


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=10, help="independent runs per instance")
    parser.add_argument("--method", choices=["abc", "bitflip"], default="abc")
    parser.add_argument("--large", action="store_true", help="include PI_1_10000 (slow)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    names = [n for n in PISINGER if args.large or n != "PI_1_10000"]
    instances = [P08] + [load_pisinger(n) for n in names]
    extra = {} if args.method == "abc" else {"max_iter": 1000}

    header = f"{'Instance':<11}{'n':>6}{'Optimum':>12}{'Best':>12}{'Mean':>14}{'Gap %':>8}{'Hit':>6}{'Time [s]':>10}"
    print(header)
    print("-" * len(header))
    for inst in instances:
        s = run_repeated(inst, runs=args.runs, seed=args.seed, method=args.method, **extra)
        print(f"{s['instance']:<11}{s['n']:>6}{s['optimum']:>12,}{s['best']:>12,.0f}{s['mean']:>14,.1f}"
              f"{s['gap_%']:>8.3f}{s['hit_rate']:>6.0%}{s['time_s']:>10.2f}")


if __name__ == "__main__":
    main()
