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
        num_initial_samples = min(20, self.budget // 3)
        initial_points = np.random.uniform(lb, ub, (num_initial_samples, self.dim))

        best_solution = None
        best_value = float('inf')
        evaluations = 0

        for point in initial_points:
            # Use Nelder-Mead for initial exploration
            res = minimize(func, point, method='Nelder-Mead',
                           options={'maxiter': max(2, (self.budget - evaluations) // num_initial_samples)})
            evaluations += res.nfev

            # Switch to BFGS if convergence stagnates or after initial exploration
            if evaluations < self.budget // 2:
                res = minimize(func, res.x, method='BFGS',
                               options={'maxiter': max(1, (self.budget - evaluations) // num_initial_samples),
                                        'gtol': 1e-5})
                evaluations += res.nfev

            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x

            if evaluations >= self.budget:
                break

        return best_solution