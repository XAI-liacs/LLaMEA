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
            # Choose optimization method dynamically
            method = 'BFGS' if np.linalg.norm(np.gradient(func(point))) > 0.1 else 'Nelder-Mead'
            
            # Local optimization using dynamic method choice
            res = minimize(func, point, method=method,
                           options={'maxiter': max(2, (self.budget - evaluations) // num_initial_samples)})  # Refined iteration limit
            evaluations += res.nfev

            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x

            # Check if we have exhausted the budget
            if evaluations >= self.budget:
                break

        return best_solution