"""Baseline: bee-inspired bit-flip local search for the 0-1 knapsack problem.

Each bee holds a binary vector x (x_i = 1 means item i is packed).
In every iteration the first ``nb`` bees of the population ("employed bees")
propose a neighbour by flipping each bit independently with probability

    p(t) = delta * (1 - t / max_iter) ** eta

and keep it only if it is strictly better (greedy acceptance). The flip
probability therefore starts at ``delta`` (broad exploration) and decays to 0
(fine exploitation). Infeasible solutions (total weight > capacity) get
fitness 0.
"""

import numpy as np


class BitFlipBee:
    """Bee-colony search for the 0-1 knapsack problem.

    Parameters
    ----------
    capacity : float
        Knapsack capacity C.
    weights, values : array-like of shape (n,)
        Item weights w_i and values v_i.
    ns : int
        Total population size (number of candidate solutions).
    nb : int
        Number of employed bees, i.e. solutions improved per iteration (nb <= ns).
    max_iter : int
        Number of iterations.
    delta : float
        Initial bit-flip probability.
    eta : float
        Decay exponent of the flip probability.
    seed : int or None
        Seed for the random number generator (None = non-reproducible).
    """

    def __init__(self, capacity, weights, values, ns, nb, max_iter,
                 delta, eta, seed=None):
        self.capacity = capacity
        self.weights = np.asarray(weights, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.num_items = len(self.weights)
        if nb > ns:
            raise ValueError("nb (employed bees) must not exceed ns (population size)")
        self.ns = ns
        self.nb = nb
        self.max_iter = max_iter
        self.delta = delta
        self.eta = eta
        self.rng = np.random.default_rng(seed)

        # Results, filled by run()
        self.best_profit = 0
        self.best_weight = 0
        self.best_combination = []
        self.convergence_progress = []   # best profit after each iteration

    # ----------------------------------------------------------------- helpers
    def dynamic_probability(self, iteration):
        """Bit-flip probability for the given iteration."""
        return self.delta * (1 - iteration / self.max_iter) ** self.eta

    def calculate_weight(self, solution):
        """Total weight of a binary solution."""
        return float(self.weights @ solution)

    def calculate_profit(self, solution):
        """Total value of a solution, or 0 if it exceeds the capacity."""
        if self.calculate_weight(solution) > self.capacity:
            return 0
        return float(self.values @ solution)

    def flip_bits_with_probability(self, solution, probability):
        """Flip each bit of ``solution`` independently with ``probability``."""
        mask = self.rng.random(self.num_items) < probability
        return np.where(mask, 1 - solution, solution)

    # -------------------------------------------------------------------- main
    def run(self):
        """Run the search and return the best profit found."""
        population = self.rng.integers(0, 2, size=(self.ns, self.num_items))
        fitness = np.array([self.calculate_profit(s) for s in population])
        self.convergence_progress = []

        for iteration in range(self.max_iter):
            probability = self.dynamic_probability(iteration)

            # Employed bees: propose a neighbour, keep it only if strictly better
            for i in range(self.nb):
                candidate = self.flip_bits_with_probability(population[i], probability)
                candidate_profit = self.calculate_profit(candidate)
                if candidate_profit > fitness[i]:
                    population[i] = candidate
                    fitness[i] = candidate_profit

            # Track the global best solution
            best = int(np.argmax(fitness))
            if fitness[best] > self.best_profit:
                self.best_profit = fitness[best]
                self.best_combination = population[best].tolist()
                self.best_weight = self.calculate_weight(population[best])

            self.convergence_progress.append(self.best_profit)

        return self.best_profit
