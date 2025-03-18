import numpy as np
from scipy.optimize import minimize

class HybridLocalGlobalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Initial uniform sampling to get diverse starting points
        num_initial_samples = min(15, self.budget // 3)
        initial_points = np.random.uniform(lb, ub, (num_initial_samples, self.dim))

        best_solution = None
        best_value = float('inf')
        evaluations = 0

        for point in initial_points:
            # Local optimization using Nelder-Mead
            res = minimize(func, point, method='Nelder-Mead',
                           options={'maxiter': max(2, (self.budget - evaluations) // num_initial_samples)})
            evaluations += res.nfev

            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x

            # Check if we have exhausted the budget
            if evaluations >= self.budget:
                break

        # Refined local optimization using BFGS for final improvement if budget allows
        if evaluations < self.budget:
            res = minimize(func, best_solution, method='BFGS', options={'maxiter': self.budget - evaluations})
            evaluations += res.nfev
            if res.fun < best_value:
                best_solution = res.x

        return best_solution