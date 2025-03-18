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

        # Enhanced initial uniform sampling
        num_initial_samples = min(25, self.budget // 4)  # Adjusted number of initial samples
        initial_points = np.random.uniform(lb, ub, (num_initial_samples, self.dim))

        best_solution = None
        best_value = float('inf')
        evaluations = 0

        for point in initial_points:
            # Dynamic adjustment of maxiter based on remaining budget
            maxiter = max(3, (self.budget - evaluations) // num_initial_samples)

            # Local optimization using BFGS with dynamic hyperparameters
            res = minimize(
                func, point, method='BFGS',
                options={'maxiter': maxiter, 'gtol': 1e-6, 'eps': 1e-8}  # Modified tol and added eps
            )
            evaluations += res.nfev

            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x

            # Check if we have exhausted the budget
            if evaluations >= self.budget:
                break

        return best_solution