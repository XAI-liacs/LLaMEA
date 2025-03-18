import numpy as np
from scipy.optimize import minimize
from pyDOE import lhs  # Importing Latin Hypercube Sampling

class HybridLocalGlobalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Initial uniform sampling to get diverse starting points (using LHS instead of uniform)
        num_initial_samples = min(15, self.budget // 3)  # Adjusted budget allocation
        initial_points = lb + (ub - lb) * lhs(self.dim, samples=num_initial_samples)  # LHS adjustment

        best_solution = None
        best_value = float('inf')
        evaluations = 0

        for point in initial_points:
            # Local optimization using BFGS
            res = minimize(func, point, method='BFGS',
                           options={'maxiter': max(2, (self.budget - evaluations) // num_initial_samples)})  # Refined iteration limit
            evaluations += res.nfev

            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x

            # Check if we have exhausted the budget
            if evaluations >= self.budget:
                break

        return best_solution