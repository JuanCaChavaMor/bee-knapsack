"""Plot convergence curves of repeated ABC runs on one instance.

Usage:
    python scripts/plot_convergence.py [--instance PI_2_500] [--runs 10] [--iters 100]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from bee_knapsack import BUILTIN, PISINGER, convergence_runs, load_pisinger, plot_convergence


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instance", default="PI_2_500", choices=list(PISINGER) + list(BUILTIN))
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--out", default="figures/convergence_{instance}.png")
    args = parser.parse_args()

    inst = BUILTIN.get(args.instance) or load_pisinger(args.instance)
    curves = convergence_runs(inst, runs=args.runs, max_iterations=args.iters)
    plot_convergence(curves, optimum=inst.optimum,
                     title=f"Discrete ABC on {inst.name} ({args.runs} runs)")

    out = Path(args.out.format(instance=inst.name))
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
