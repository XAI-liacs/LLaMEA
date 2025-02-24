import numpy as np
from scipy.optimize import minimize, differential_evolution

class MetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract the bounds and prepare for optimizations
        lower_bounds = func.bounds.lb
        upper_bounds = func.bounds.ub
        bounds = [(low, high) for low, high in zip(lower_bounds, upper_bounds)]
        
        # Calculate the number of initial differential evolution samples based on the available budget
        num_de_samples = max(self.budget // 2, 5)
        remaining_budget = self.budget - num_de_samples

        # Step 1: Use Differential Evolution for global exploration
        def wrapped_func_de(x):
            nonlocal remaining_budget
            if remaining_budget <= 0:
                return float('inf')
            remaining_budget -= 1
            return func(x)

        result_de = differential_evolution(wrapped_func_de, bounds, maxiter=100, popsize=num_de_samples, tol=0.01)
        best_solution = result_de.x
        best_score = result_de.fun
        
        # Step 2: Use BFGS local optimization from the best DE solution
        def wrapped_func(x):
            nonlocal remaining_budget
            if remaining_budget <= 0:
                return float('inf')
            remaining_budget -= 1
            return func(x)

        result = minimize(wrapped_func, best_solution, method='L-BFGS-B', bounds=bounds, options={'maxfun': remaining_budget, 'ftol': 1e-9})

        # Return the best found solution
        return result.x if result.success else best_solution