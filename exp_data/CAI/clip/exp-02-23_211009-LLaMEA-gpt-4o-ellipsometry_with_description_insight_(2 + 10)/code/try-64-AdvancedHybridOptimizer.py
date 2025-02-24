import numpy as np
from scipy.optimize import minimize

class AdvancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub

        # Adjust initial sampling points with a focus on balance between exploration and exploitation
        initial_samples = max(min(self.budget // (3 * self.dim), 80), 15)
        remaining_budget = self.budget - initial_samples

        # Uniformly sample initial points using refined strategy
        samples = np.random.uniform(lb, ub, (initial_samples, self.dim))
        best_value = float('inf')
        best_solution = None

        # Evaluate sampled points
        evaluations = 0
        for sample in samples:
            if evaluations >= self.budget:
                break
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample

        # Define a bounded function to ensure the search remains within the specified bounds
        def bounded_func(x):
            return func(np.clip(x, lb, ub))

        # Use the remaining budget efficiently in local optimization with adaptive L-BFGS-B
        # Increased precision for adaptive convergence
        options = {'maxiter': remaining_budget, 'disp': False, 'gtol': 1e-7}
        result = minimize(bounded_func, best_solution, method='L-BFGS-B', bounds=np.array([lb, ub]).T, options=options)

        return result.x