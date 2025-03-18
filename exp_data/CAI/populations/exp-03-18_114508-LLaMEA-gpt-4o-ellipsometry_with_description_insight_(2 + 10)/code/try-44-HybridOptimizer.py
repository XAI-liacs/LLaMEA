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

        # Adjust initial sampling size based on remaining budget
        num_initial_samples = max(1, (self.budget - self.evaluations) // 8)
        initial_samples = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=(num_initial_samples, self.dim))

        for sample in initial_samples:
            if self.evaluations >= self.budget:
                break
            solution, value = self.local_search(func, sample, bounds, dynamic=True)  # Added dynamic flag
            if value < best_value:
                best_solution, best_value = solution, value
                bounds = self.tighten_bounds(bounds, best_solution, factor=0.1)  # Tighten bounds around best solution

        return best_solution

    def local_search(self, func, initial_point, bounds, dynamic=False):
        if self.evaluations >= self.budget:
            return initial_point, func(initial_point)

        # Use a local optimizer (BFGS) for fast convergence, dynamic bound tightening
        options = {'maxfun': self.budget - self.evaluations}
        if dynamic:
            options['maxiter'] = 50  # Dynamic setting for more iterative control
        result = minimize(func, initial_point, method='L-BFGS-B', bounds=bounds, options=options)
        self.evaluations += result.nfev

        return result.x, result.fun

    def tighten_bounds(self, bounds, best_solution, factor=0.1):
        # A function to tighten the search bounds around the best solution
        new_bounds = np.empty_like(bounds)
        for i in range(self.dim):
            range_size = factor * (bounds[i, 1] - bounds[i, 0])
            new_bounds[i] = [
                max(bounds[i, 0], best_solution[i] - range_size / 2),
                min(bounds[i, 1], best_solution[i] + range_size / 2)
            ]
        return new_bounds