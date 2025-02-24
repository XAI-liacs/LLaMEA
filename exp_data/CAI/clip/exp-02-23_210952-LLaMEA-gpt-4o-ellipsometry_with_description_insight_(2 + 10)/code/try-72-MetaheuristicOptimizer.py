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
        
        # Adaptive sampling strategy: Adjust initial samples based on budget and dimension
        num_initial_samples = max(min(self.budget // (self.dim * 2), 10), 5)
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

        # Dynamic adjustment of BFGS options for better convergence with early stopping criterion
        bfgs_options = {
            'maxfun': remaining_budget,
            'ftol': 1e-9,
            'gtol': 1e-6 / self.dim,
            'eps': 1e-8  # Early stopping criterion based on minimal improvement (modified line)
        }

        result = minimize(wrapped_func, best_solution, method='L-BFGS-B', bounds=bounds, options=bfgs_options)

        # Return the best found solution
        return result.x if result.success else best_solution