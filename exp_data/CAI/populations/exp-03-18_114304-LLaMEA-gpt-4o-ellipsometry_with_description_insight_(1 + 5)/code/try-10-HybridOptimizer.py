import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds from the function
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')
        evaluations = 0

        # Initial exploration with Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=self.dim)
        samples = qmc.scale(sampler.random(n=self.budget // 2), lb, ub)
        for sample in samples:
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample

        # Local exploitation using BFGS starting from the best sample found
        def local_optimization(x):
            nonlocal evaluations
            if evaluations >= self.budget:
                return best_value  # Return the best value found if budget is exhausted
            value = func(x)
            evaluations += 1
            return value

        # Update bounds dynamically for tighter constraints
        dynamic_bounds = [(max(lb[i], best_solution[i] - 0.1), min(ub[i], best_solution[i] + 0.1)) for i in range(self.dim)]
        
        # Run BFGS from the best initial sample
        result = minimize(local_optimization, best_solution, method='BFGS',
                          bounds=dynamic_bounds,
                          options={'maxiter': self.budget - evaluations, 'disp': False})

        if result.fun < best_value:
            best_value = result.fun
            best_solution = result.x

        return best_solution, best_value