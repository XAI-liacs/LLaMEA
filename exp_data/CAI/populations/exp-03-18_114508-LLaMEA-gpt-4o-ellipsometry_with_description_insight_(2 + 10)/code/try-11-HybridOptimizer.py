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

        num_initial_samples = max(1, self.budget // 10)
        initial_samples = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=(num_initial_samples, self.dim))

        for sample in initial_samples:
            if self.evaluations >= self.budget:
                break
            solution, value = self.local_search(func, sample, bounds)
            if value < best_value:
                best_solution, best_value = solution, value

        # Restart mechanism to further explore if budget allows
        while self.evaluations + 10 <= self.budget:
            random_sample = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=self.dim)
            solution, value = self.local_search(func, random_sample, bounds)
            if value < best_value:
                best_solution, best_value = solution, value

        return best_solution

    def local_search(self, func, initial_point, bounds):
        if self.evaluations >= self.budget:
            return initial_point, func(initial_point)

        result = minimize(func, initial_point, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - self.evaluations})
        self.evaluations += result.nfev

        return result.x, result.fun