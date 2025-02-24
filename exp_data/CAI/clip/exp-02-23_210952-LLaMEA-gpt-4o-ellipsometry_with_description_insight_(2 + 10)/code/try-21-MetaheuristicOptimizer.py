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
        
        # Dynamic initial sampling based on budget
        num_initial_samples = max(1, min(self.budget // 4, 8))
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
        
        # Step 2: Use BFGS local optimization from the best initial samples
        def wrapped_func(x):
            nonlocal remaining_budget
            if remaining_budget <= 0:
                return float('inf')
            remaining_budget -= 1
            return func(x)

        # Try optimization and restart if stuck
        while remaining_budget > 0:
            result = minimize(wrapped_func, best_solution, method='L-BFGS-B', bounds=bounds, options={'maxfun': remaining_budget})
            if result.success and result.fun < best_score:
                best_score = result.fun
                best_solution = result.x
            else:
                initial_solutions = np.random.uniform(lower_bounds, upper_bounds, (1, self.dim))
                best_solution = initial_solutions[0]

        # Return the best found solution
        return best_solution