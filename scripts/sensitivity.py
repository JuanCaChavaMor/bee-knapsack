"""Sensitivity of the discrete ABC to the colony sizes (mean over repeated runs).

Usage:
    python scripts/sensitivity.py [--instance PI_2_100] [--runs 10]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from bee_knapsack import load_pisinger, sensitivity

GRID = {
    "num_employed_bees": range(10, 101, 10),
    "num_onlooker_bees": range(10, 101, 10),
    "num_elite_bees": range(2, 10),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instance", default="PI_2_100")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--out", default="figures/sensitivity_{instance}.png")
    args = parser.parse_args()

    inst = load_pisinger(args.instance)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex="col", sharey="row")
    for col, (param, values) in enumerate(GRID.items()):
        rows = sensitivity(inst, param, values, runs=args.runs)
        x = [r[param] for r in rows]
        axes[0, col].plot(x, [r["mean"] for r in rows], marker="o")
        axes[0, col].axhline(inst.optimum, color="black", linestyle="--", linewidth=1)
        axes[0, col].set_title(param)
        axes[1, col].plot(x, [r["time_s"] for r in rows], marker="o", color="tab:orange")
        axes[1, col].set_xlabel(param)
        print(param, [(r[param], round(r["mean"], 1), r["hit_rate"]) for r in rows])
    axes[0, 0].set_ylabel(f"Mean profit ({args.runs} runs)")
    axes[1, 0].set_ylabel("Mean time per run [s]")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Sensitivity of the discrete ABC on {inst.name}")
    fig.tight_layout()

    out = Path(args.out.format(instance=inst.name))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
