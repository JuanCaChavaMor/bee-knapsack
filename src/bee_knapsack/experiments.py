"""Helpers to run, compare and visualise the algorithms on an Instance."""

import itertools
import time

import matplotlib.pyplot as plt
import numpy as np

from .abc import ArtificialBeeColony
from .baseline import BitFlipBee


# ------------------------------------------------------------------ factories
def make_abc(instance, seed=None, **params):
    """Discrete ABC with the parameters of Sabet et al. (2012), overridable."""
    return ArtificialBeeColony(instance.weights, instance.values, instance.capacity,
                               seed=seed, **params)


def make_bee(instance, max_iter=1000, delta=0.5, eta=0.9, ns=None, nb=None, seed=None):
    """Baseline bit-flip search with the default sizing ns = 2n, nb = n."""
    n = instance.num_items
    return BitFlipBee(instance.capacity, instance.weights, instance.values,
                      ns=ns or 2 * n, nb=nb or n, max_iter=max_iter,
                      delta=delta, eta=eta, seed=seed)


def _solve(algo):
    """Run either algorithm and return (best_profit, convergence curve)."""
    if isinstance(algo, ArtificialBeeColony):
        algo.optimize()
        return algo.best_profit, algo.convergence
    algo.run()
    return algo.best_profit, algo.convergence_progress


def relative_error(profit, optimum):
    """Gap to the known optimum in percent."""
    return 100 * abs(optimum - profit) / optimum


# ---------------------------------------------------------------- experiments
def run_repeated(instance, runs=10, seed=0, method="abc", **params):
    """Run ``runs`` independent repetitions and summarise them.

    Returns a dict with best/mean profit, hit rate (fraction of runs that
    reached the known optimum), mean gap in percent and mean runtime.
    """
    factory = make_abc if method == "abc" else make_bee
    profits, times = [], []
    for r in range(runs):
        algo = factory(instance, seed=seed + r, **params)
        t0 = time.perf_counter()
        profit, _ = _solve(algo)
        times.append(time.perf_counter() - t0)
        profits.append(profit)
    profits = np.array(profits)
    summary = {"instance": instance.name, "n": instance.num_items,
               "optimum": instance.optimum, "best": profits.max(),
               "mean": profits.mean(), "time_s": float(np.mean(times))}
    if instance.optimum:
        summary["hit_rate"] = float(np.mean(np.isclose(profits, instance.optimum)))
        summary["gap_%"] = relative_error(profits.mean(), instance.optimum)
    return summary


def sensitivity(instance, param, values, runs=10, seed=0, **fixed):
    """Vary one ABC parameter and record mean profit and runtime.

    Unlike a single run per value, averaging over ``runs`` repetitions
    separates the effect of the parameter from random fluctuations.
    """
    rows = []
    for v in values:
        s = run_repeated(instance, runs=runs, seed=seed, **{**fixed, param: v})
        rows.append({param: v, "mean": s["mean"], "hit_rate": s.get("hit_rate"),
                     "time_s": s["time_s"]})
    return rows


def parameter_sweep(instance, etas, deltas, runs=10, max_iter=1000, seed=0):
    """Baseline only: grid search over (eta, delta) with repeated runs."""
    rows = []
    for eta, delta in itertools.product(etas, deltas):
        s = run_repeated(instance, runs=runs, seed=seed, method="bitflip",
                         max_iter=max_iter, delta=delta, eta=eta)
        rows.append({"eta": eta, "delta": delta, **{k: s[k] for k in
                     ("mean", "best", "hit_rate", "gap_%")}})
    return sorted(rows, key=lambda r: r["mean"], reverse=True)


def convergence_runs(instance, runs=20, seed=0, method="abc", **params):
    """Convergence curves (best profit per iteration) of independent runs."""
    factory = make_abc if method == "abc" else make_bee
    return [_solve(factory(instance, seed=seed + r, **params))[1] for r in range(runs)]


# ---------------------------------------------------------------------- plots
def plot_convergence(curves, optimum=None, title="Convergence", ax=None):
    """Plot best profit vs. iteration for several runs."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    for curve in curves:
        ax.plot(np.arange(1, len(curve) + 1), curve, linewidth=1, alpha=0.7)
    if optimum is not None:
        ax.axhline(optimum, color="black", linestyle="--", linewidth=1, label="Optimum")
        ax.legend()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best profit")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax
