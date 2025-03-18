import numpy as np
from scipy.optimize import minimize
from skopt import Optimizer

class MetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract the bounds and prepare for optimizations
        lower_bounds = func.bounds.lb
        upper_bounds = func.bounds.ub
        bounds = [(low, high) for low, high in zip(lower_bounds, upper_bounds)]
        
        # Bayesian Optimization for adaptive sampling
        opt = Optimizer(dimensions=bounds, n_initial_points=min(self.budget // (self.dim * 2), 10))
        
        remaining_budget = self.budget - len(opt.Xi)
        
        # Initialize the best solution found so far
        best_solution = None
        best_score = float('inf')

        # Step 1: Use Bayesian Optimization for initial solutions
        for _ in range(len(opt.Xi)):
            solution = opt.ask()
            score = func(solution)
            opt.tell(solution, score)
            if score < best_score:
                best_score = score
                best_solution = solution
        
        # Step 2: Use L-BFGS-B local optimization from the best initial samples
        def wrapped_func(x):
            nonlocal remaining_budget
            if remaining_budget <= 0:
                return float('inf')
            remaining_budget -= 1
            return func(x)

        # Dynamic adjustment of L-BFGS-B options for better convergence
        bfgs_options = {
            'maxfun': remaining_budget,
            'ftol': 1e-8,
            'gtol': 1e-7 / self.dim
        }

        result = minimize(wrapped_func, best_solution, method='L-BFGS-B', bounds=bounds, options=bfgs_options)

        # Return the best found solution
        return result.x if result.success else best_solution