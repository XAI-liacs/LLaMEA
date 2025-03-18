import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.current_evaluations = 0

    def differential_evolution(self, func, bounds, pop_size=20, F=0.8, CR=0.9):
        pop = np.random.uniform(bounds.lb, bounds.ub, (pop_size, self.dim))
        best_idx = np.argmin([func(ind) for ind in pop])
        best = pop[best_idx]
        self.current_evaluations += pop_size

        while self.current_evaluations < self.budget:
            for i in range(pop_size):
                if self.current_evaluations >= self.budget:
                    break

                indices = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                self.current_evaluations += 1

                if trial_fitness < func(pop[i]):
                    pop[i] = trial
                    if trial_fitness < func(best):
                        best = trial

        return best

    def local_search(self, func, x0, bounds):
        x0 = x0 - np.sign(np.gradient(func(x0))) * 0.01  # Line changed: Initial guess adjustment
        result = minimize(func, x0, bounds=bounds, method='L-BFGS-B')
        return result.x

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        best_de = self.differential_evolution(func, func.bounds)

        if self.current_evaluations < self.budget:
            best_solution = self.local_search(func, best_de, bounds)
        else:
            best_solution = best_de

        return best_solution