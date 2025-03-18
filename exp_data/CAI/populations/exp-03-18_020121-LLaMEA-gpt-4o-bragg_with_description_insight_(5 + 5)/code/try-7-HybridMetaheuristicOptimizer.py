import numpy as np

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Heuristic for population size
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.population = None
        self.best_solution = None
        self.best_cost = float('inf')
        self.bounds = None

    def initialize_population(self, lower_bound, upper_bound):
        # Quasi-Oppositional Initialization
        self.population = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        self.population = np.concatenate((self.population, lower_bound + upper_bound - self.population[:self.population_size//2]), axis=0)

    def evaluate_population(self, func):
        costs = np.array([func(ind) for ind in self.population])
        best_idx = np.argmin(costs)
        if costs[best_idx] < self.best_cost:
            self.best_cost = costs[best_idx]
            self.best_solution = self.population[best_idx].copy()
        return costs

    def differential_evolution(self, func):
        for _ in range(self.budget):
            for i in range(self.population_size):
                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = self.population[np.random.choice(candidates, 3, replace=False)]
                mutant_vector = np.clip(a + self.mutation_factor * (b - c), func.bounds.lb, func.bounds.ub)
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])
                trial_cost = func(trial_vector)
                if trial_cost < func(self.population[i]):
                    self.population[i] = trial_vector
                    if trial_cost < self.best_cost:
                        self.best_cost = trial_cost
                        self.best_solution = trial_vector.copy()

    def local_search(self, func):
        from scipy.optimize import minimize
        res = minimize(func, self.best_solution, method='trust-constr', bounds=[(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)])
        if res.fun < self.best_cost:
            self.best_cost = res.fun
            self.best_solution = res.x

    def __call__(self, func):
        self.bounds = func.bounds
        self.initialize_population(func.bounds.lb, func.bounds.ub)
        self.budget -= self.population_size
        self.evaluate_population(func)
        self.differential_evolution(func)
        if self.budget > 0:
            self.local_search(func)
        return self.best_solution, self.best_cost