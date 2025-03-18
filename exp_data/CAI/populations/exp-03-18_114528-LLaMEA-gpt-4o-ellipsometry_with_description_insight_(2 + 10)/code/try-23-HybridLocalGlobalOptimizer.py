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
        num_initial_samples = min(15, self.budget // 3)  # Adjusted budget allocation
        initial_points = np.random.uniform(lb, ub, (num_initial_samples, self.dim))

        best_solution = None
        best_value = float('inf')
        evaluations = 0

        for point in initial_points:
            # Local optimization using Nelder-Mead followed by BFGS for refinement
            res = minimize(func, point, method='Nelder-Mead',
                           options={'maxiter': max(2, (self.budget - evaluations) // num_initial_samples)})  # Refined iteration limit
            # Switch to BFGS for improved precision
            if evaluations < self.budget:
                res_bfgs = minimize(func, res.x, method='BFGS',
                                    options={'maxiter': max(2, (self.budget - evaluations) // num_initial_samples)})
                evaluations += res_bfgs.nfev

                if res_bfgs.fun < best_value:
                    best_value = res_bfgs.fun
                    best_solution = res_bfgs.x

            # Check if we have exhausted the budget
            if evaluations >= self.budget:
                break

        return best_solution