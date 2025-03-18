import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        best_solution = None
        best_value = float('inf')

        num_initial_samples = max(1, (self.budget - self.evaluations) // 8)
        initial_samples = np.random.normal(loc=(func.bounds.lb + func.bounds.ub) / 2,
                                           scale=(func.bounds.ub - func.bounds.lb) / 6,
                                           size=(num_initial_samples, self.dim))

        for sample in initial_samples:
            if self.evaluations >= self.budget:
                break
            new_bounds = np.clip(bounds, func.bounds.lb, func.bounds.ub)
            solution, value = self.local_search(func, sample, new_bounds)
            if value < best_value:
                best_solution, best_value = solution, value

        return best_solution

    def local_search(self, func, initial_point, bounds):
        if self.evaluations >= self.budget:
            return initial_point, func(initial_point)

        result = minimize(func, initial_point, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - self.evaluations})
        self.evaluations += result.nfev

        return result.x, result.fun