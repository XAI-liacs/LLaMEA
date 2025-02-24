import numpy as np
from scipy.optimize import minimize

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
        num_initial_samples = max(int(self.budget * 0.2), 5)  # Changed line
        remaining_budget = self.budget - num_initial_samples

        # Initialize the best solution found so far
        best_solution = None
        best_score = float('inf')

        # Step 1: Uniformly sample the initial solutions
        initial_solutions = np.random.uniform(lower_bounds, upper_bounds, (num_initial_samples, self.dim))
        
        for solution in initial_solutions:
            score = func(solution)
            if score < best_score:
                best_score = score
                best_solution = solution

        # Adaptive budget allocation for BFGS based on initial solution scores
        adjusted_remaining_budget = remaining_budget // 2 if best_score > 1e-5 else remaining_budget  # Changed line
        
        # Step 2: Use BFGS local optimization from the best initial samples
        def wrapped_func(x):
            nonlocal adjusted_remaining_budget
            if adjusted_remaining_budget <= 0:
                return float('inf')
            adjusted_remaining_budget -= 1  # Changed line
            return func(x)

        result = minimize(wrapped_func, best_solution, method='L-BFGS-B', bounds=bounds, options={'maxfun': adjusted_remaining_budget, 'ftol': 1e-9})  # Changed line

        # Return the best found solution
        return result.x if result.success else best_solution