import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.used_budget = 0

    def differential_evolution(self, func, bounds, population_size=50, F=0.5, CR=0.9):
        population = np.random.uniform(bounds.lb, bounds.ub, (population_size, self.dim))
        best_idx = np.argmin([func(ind) for ind in population])
        best = population[best_idx]

        while self.used_budget < self.budget:
            for i in range(population_size):
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
                CR = 0.5 + 0.4 * (self.budget - self.used_budget) / self.budget  # Adaptive crossover rate
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                f = func(trial)
                self.used_budget += 1
                if f < func(population[i]):
                    population[i] = trial
                    if f < func(best):
                        best = trial
            if self.used_budget >= 0.5 * self.budget:
                break
        return best

    def bfgs_refinement(self, func, x0, bounds):
        result = minimize(func, x0, method='L-BFGS-B', bounds=[(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)])
        self.used_budget += result.nfev
        return result.x, result.fun

    def __call__(self, func):
        bounds = func.bounds
        # Global exploration with symmetry
        candidate = self.differential_evolution(func, bounds)
        # Local refinement
        final_solution, final_value = self.bfgs_refinement(func, candidate, bounds)
        return final_solution, final_value