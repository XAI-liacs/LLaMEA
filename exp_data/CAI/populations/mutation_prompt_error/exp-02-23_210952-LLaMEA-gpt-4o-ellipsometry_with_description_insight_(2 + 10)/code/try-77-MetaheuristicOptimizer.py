import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class MetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract the bounds and prepare for optimizations
        lower_bounds = func.bounds.lb
        upper_bounds = func.bounds.ub
        bounds = [(low, high) for low, high in zip(lower_bounds, upper_bounds)]
        
        # Adaptive sampling strategy: Adjust initial samples based on budget and dimension
        num_initial_samples = max(min(self.budget // (self.dim * 2), 10), 5)
        remaining_budget = self.budget - num_initial_samples

        # Initialize the best solution found so far
        best_solution = None
        best_score = float('inf')

        # Step 1: Use Latin Hypercube Sampling for initial solutions
        sampler = qmc.LatinHypercube(d=self.dim)
        initial_solutions = qmc.scale(sampler.random(n=num_initial_samples), lower_bounds, upper_bounds)

        for solution in initial_solutions:
            score = func(solution)
            if score < best_score:
                best_score = score
                best_solution = solution
        
        # Step 2: Use BFGS local optimization from the best initial samples
        def wrapped_func(x):
            nonlocal remaining_budget
            if remaining_budget <= 0:
                return float('inf')
            remaining_budget -= 1
            return func(x)

        # Dynamic adjustment of BFGS options for better convergence
        bfgs_options = {
            'maxfun': remaining_budget,
            'ftol': 1e-9,
            'gtol': 1e-6 / self.dim
        }

        result = minimize(wrapped_func, best_solution, method='L-BFGS-B', bounds=bounds, options=bfgs_options)

        # Return the best found solution
        return result.x if result.success else best_solution