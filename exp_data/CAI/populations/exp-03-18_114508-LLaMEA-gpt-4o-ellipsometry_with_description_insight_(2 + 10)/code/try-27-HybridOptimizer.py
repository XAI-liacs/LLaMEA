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

        # Adaptive sampling: Adjust number of initial samples based on dimensionality
        num_initial_samples = max(5, self.budget // (5 * self.dim))
        initial_samples = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=(num_initial_samples, self.dim))

        for sample in initial_samples:
            if self.evaluations >= self.budget:
                break
            solution, value = self.local_search(func, sample, bounds)
            if value < best_value:
                best_solution, best_value = solution, value

        return best_solution

    def local_search(self, func, initial_point, bounds):
        if self.evaluations >= self.budget:
            return initial_point, func(initial_point)

        # Use L-BFGS-B with tighter convergence criteria
        result = minimize(func, initial_point, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - self.evaluations, 'ftol': 1e-9})
        self.evaluations += result.nfev

        return result.x, result.fun