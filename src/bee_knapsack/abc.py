"""Discrete Artificial Bee Colony (ABC) for the 0-1 knapsack problem.

Implementation of the binary ABC of Sabet, Farokhi and Shokouhifar (2012),
"A Novel Artificial Bee Colony Algorithm for the Knapsack Problem".

Algorithm outline (one iteration):

1. Evaluate the employed bees (each one holds a feasible binary solution).
2. Onlooker selection: each onlooker picks an employed bee k with probability
   proportional to Ch_k = (F_k / sum_i F_i) ** mu (elitist for mu >> 1).
3. Recruitment: the selected bee k gets S_k = int(beta * Ch_k / H + gamma)
   "soldiers" (beta < 0, gamma > 0), i.e. neighbourhood trials.
4. Neighbourhood search: each soldier flips every bit with probability
       p_change = p_min + (t / t_max) * (p_max - p_min)
   and repairs infeasible solutions by removing the packed items with the
   lowest value-to-weight ratio until the capacity holds. The trial replaces
   the bee only if it is strictly better.
5. Record the global best solution.

Initialisation is greedy: items are packed in decreasing order of v_i / w_i
while they fit. All employed bees start from this same solution, so the
diversity of the colony comes entirely from the neighbourhood search.
"""

import numpy as np


class ArtificialBeeColony:
    """Binary ABC for the 0-1 knapsack problem.

    Parameters
    ----------
    weights, profits : array-like of shape (n,)
        Item weights w_i and profits v_i.
    capacity : float
        Knapsack capacity C.
    num_employed_bees : int
        Number of food sources (candidate solutions).
    num_onlooker_bees : int
        Number of onlookers, i.e. selections per iteration.
    num_elite_bees : int
        H in the recruitment formula (number of elite bees).
    mu : float
        Elitism exponent of the onlooker selection.
    beta, gamma : float
        Recruitment parameters (beta < 0, gamma > 0).
    p_change_min, p_change_max : float
        Bit-flip probability at the first and last iteration.
    max_iterations : int
        Number of iterations.
    seed : int or None
        Seed of the random number generator.
    """

    def __init__(self, weights, profits, capacity, num_employed_bees=50,
                 num_onlooker_bees=50, num_elite_bees=5, mu=3, beta=-10,
                 gamma=16, p_change_min=0.04, p_change_max=0.12,
                 max_iterations=100, seed=None):
        self.weights = np.asarray(weights, dtype=float)
        self.profits = np.asarray(profits, dtype=float)
        self.capacity = capacity
        self.num_items = len(self.weights)
        self.num_employed_bees = num_employed_bees
        self.num_onlooker_bees = num_onlooker_bees
        self.H = num_elite_bees
        self.mu = mu
        self.beta = beta
        self.gamma = gamma
        self.p_change_min = p_change_min
        self.p_change_max = p_change_max
        self.max_iterations = max_iterations
        self.rng = np.random.default_rng(seed)

        # Value-to-weight ratios (0 for zero-weight items, avoids division by 0)
        self.ratios = np.divide(self.profits, self.weights,
                                out=np.zeros_like(self.profits),
                                where=self.weights != 0)

        initial = self.greedy_solution()
        self.employed_bees = [initial.copy() for _ in range(num_employed_bees)]
        self.best_solution = initial.copy()
        self.best_profit = self.calculate_profit(initial)
        self.convergence = []   # best profit after each iteration

    # ----------------------------------------------------------------- helpers
    def greedy_solution(self):
        """Pack items by decreasing value-to-weight ratio while they fit."""
        solution = np.zeros(self.num_items, dtype=bool)
        total_weight = 0.0
        for i in np.argsort(-self.ratios):
            if total_weight + self.weights[i] <= self.capacity:
                solution[i] = True
                total_weight += self.weights[i]
        return solution

    def calculate_weight(self, solution):
        return float(self.weights @ solution)

    def calculate_profit(self, solution):
        return float(self.profits @ solution)

    def repair(self, solution):
        """Drop the packed items with the lowest ratio until the solution fits."""
        total_weight = self.calculate_weight(solution)
        if total_weight <= self.capacity:
            return solution
        packed = np.flatnonzero(solution)
        for i in packed[np.argsort(self.ratios[packed])]:
            solution[i] = False
            total_weight -= self.weights[i]
            if total_weight <= self.capacity:
                break
        return solution

    # ------------------------------------------------------------ ABC phases
    def select_onlookers(self, profits):
        """Roulette selection with probabilities proportional to Ch_k."""
        chance = (profits / profits.sum()) ** self.mu
        cumulative = np.cumsum(chance / chance.sum())
        picks = np.searchsorted(cumulative, self.rng.random(self.num_onlooker_bees))
        return np.minimum(picks, len(profits) - 1)   # guard against round-off

    def recruit_soldiers(self, onlookers, profits):
        """Number of neighbourhood trials S_k for every selected bee."""
        chance = (profits / profits.sum()) ** self.mu
        return {k: int(self.beta * chance[k] / self.H + self.gamma) for k in set(onlookers)}

    def search_neighborhood(self, solution, iteration):
        """Flip bits with the iteration-dependent probability, then repair."""
        p_change = self.p_change_min + (self.p_change_max - self.p_change_min) * (
            iteration / self.max_iterations)
        flip = self.rng.random(self.num_items) < p_change
        return self.repair(np.logical_xor(solution, flip))

    # -------------------------------------------------------------------- main
    def optimize(self):
        """Run the algorithm. Returns (best_solution, best_profit)."""
        self.convergence = []
        for iteration in range(self.max_iterations):
            profits = np.array([self.calculate_profit(s) for s in self.employed_bees])
            onlookers = self.select_onlookers(profits)
            soldiers = self.recruit_soldiers(onlookers, profits)

            for k in onlookers:
                for _ in range(soldiers[k]):
                    candidate = self.search_neighborhood(self.employed_bees[k], iteration)
                    candidate_profit = self.calculate_profit(candidate)
                    if candidate_profit > profits[k]:
                        self.employed_bees[k] = candidate
                        profits[k] = candidate_profit

            best = int(np.argmax(profits))
            if profits[best] > self.best_profit:
                self.best_profit = float(profits[best])
                self.best_solution = self.employed_bees[best].copy()
            self.convergence.append(self.best_profit)

        return self.best_solution, self.best_profit

    @staticmethod
    def to_binary_string(solution):
        return "".join("1" if x else "0" for x in solution)
