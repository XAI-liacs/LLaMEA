import numpy as np
from scipy.optimize import minimize
from pyDOE2 import lhs  # Added line for LHS

class MetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract the bounds and prepare for optimizations
        lower_bounds = func.bounds.lb
        upper_bounds = func.bounds.ub
        bounds = [(low, high) for low, high in zip(lower_bounds, upper_bounds)]
        
        # Calculate the number of initial samples based on the available budget
        num_initial_samples = max(self.budget // 3, 5)
        remaining_budget = self.budget - num_initial_samples

        # Initialize the best solution found so far
        best_solution = None
        best_score = float('inf')

        # Step 1: Use Latin Hypercube Sampling for initial solutions
        lhs_samples = lhs(self.dim, samples=num_initial_samples)  # Changed line
        initial_solutions = lower_bounds + (upper_bounds - lower_bounds) * lhs_samples  # Changed line
        
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

        # Adjust bounds dynamically for the BFGS step
        adjusted_bounds = [(max(low, best_solution[i] - 0.1 * (high - low)), min(high, best_solution[i] + 0.1 * (high - low))) for i, (low, high) in enumerate(bounds)]  # Changed line
        result = minimize(wrapped_func, best_solution, method='L-BFGS-B', bounds=adjusted_bounds, options={'maxfun': remaining_budget, 'ftol': 1e-9})  # Changed line

        # Return the best found solution
        return result.x if result.success else best_solution