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
        num_initial_samples = min(10, self.budget // 4)  # Adjust budget allocation
        initial_points = np.random.uniform(lb, ub, (num_initial_samples, self.dim))

        best_solution = None
        best_value = float('inf')
        evaluations = 0

        for point in initial_points:
            # Dynamic choice of optimization method
            method = 'BFGS' if self.dim == 2 else 'Nelder-Mead'
            # Local optimization using selected method
            res = minimize(func, point, method=method,
                           bounds=[(lb[i], ub[i]) for i in range(self.dim)],
                           options={'maxiter': max(1, (self.budget - evaluations) // num_initial_samples)})
            evaluations += res.nfev

            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x

            # Check if we have exhausted the budget
            if evaluations >= self.budget:
                break

        return best_solution