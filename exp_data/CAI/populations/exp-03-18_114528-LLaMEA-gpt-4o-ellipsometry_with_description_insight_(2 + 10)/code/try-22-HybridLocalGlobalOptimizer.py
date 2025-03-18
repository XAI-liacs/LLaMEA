import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc  # Import Latin Hypercube Sampling

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

        # Use Latin Hypercube Sampling instead of uniform sampling
        sampler = qmc.LatinHypercube(d=self.dim)
        initial_points = qmc.scale(sampler.random(num_initial_samples), lb, ub)

        best_solution = None
        best_value = float('inf')
        evaluations = 0

        for point in initial_points:
            # Local optimization using Nelder-Mead
            res = minimize(func, point, method='Nelder-Mead',
                           options={'maxiter': max(2, (self.budget - evaluations) // num_initial_samples)})  # Refined iteration limit
            evaluations += res.nfev

            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x

            # Check if we have exhausted the budget
            if evaluations >= self.budget:
                break

        return best_solution