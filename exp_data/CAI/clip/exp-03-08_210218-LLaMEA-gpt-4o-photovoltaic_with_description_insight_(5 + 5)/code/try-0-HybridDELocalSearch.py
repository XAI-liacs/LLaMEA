import numpy as np
from scipy.optimize import minimize

class HybridDELocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.f = 0.7  # Differential weight
        self.cr = 0.9  # Crossover probability

    def differential_evolution(self, func, bounds):
        pop = np.random.rand(self.population_size, self.dim)
        pop = bounds.lb + pop * (bounds.ub - bounds.lb)
        fitness = np.apply_along_axis(func, 1, pop)
        budget_used = self.population_size

        while budget_used < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = pop[indices]
                mutant = np.clip(x0 + self.f * (x1 - x2), bounds.lb, bounds.ub)
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                trial_fitness = func(trial)
                budget_used += 1
                if trial_fitness < fitness[i]:
                    pop[i], fitness[i] = trial, trial_fitness

                if budget_used >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]

    def local_search(self, func, x, bounds):
        def penalty_func(x):
            penalty = 0
            if np.any(x < bounds.lb) or np.any(x > bounds.ub):
                penalty = np.sum((np.minimum(x - bounds.lb, 0) ** 2) + 
                                 (np.maximum(x - bounds.ub, 0) ** 2))
            return func(x) + penalty

        res = minimize(penalty_func, x, bounds=[(l, u) for l, u in zip(bounds.lb, bounds.ub)],
                       method='L-BFGS-B', options={'maxfun': self.budget - self.budget_used})
        return res.x, res.fun

    def __call__(self, func):
        bounds = func.bounds
        # First stage: Differential Evolution
        best_solution, best_fitness = self.differential_evolution(func, bounds)
        self.budget_used = self.budget - self.local_search_budget
        # Second stage: Local Search
        best_solution, best_fitness = self.local_search(func, best_solution, bounds)
        return best_solution