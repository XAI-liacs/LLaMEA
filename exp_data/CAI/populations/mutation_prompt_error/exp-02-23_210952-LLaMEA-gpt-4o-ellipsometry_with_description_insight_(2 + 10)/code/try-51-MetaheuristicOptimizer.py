import numpy as np
from scipy.optimize import minimize

class MetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lower_bounds = func.bounds.lb
        upper_bounds = func.bounds.ub
        bounds = [(low, high) for low, high in zip(lower_bounds, upper_bounds)]
        
        num_initial_samples = max(min(self.budget // (self.dim * 2), 15), 5)  # Adjusted max samples
        remaining_budget = self.budget - num_initial_samples

        best_solution = None
        best_score = float('inf')

        initial_solutions = np.random.uniform(lower_bounds, upper_bounds, (num_initial_samples, self.dim))
        
        for solution in initial_solutions:
            score = func(solution)
            if score < best_score:
                best_score = score
                best_solution = solution
        
        def wrapped_func(x):
            nonlocal remaining_budget
            if remaining_budget <= 0:
                return float('inf')
            remaining_budget -= 1
            return func(x)

        bfgs_options = {
            'maxfun': remaining_budget,
            'ftol': 1e-8,  # Adjusted tolerance
            'gtol': 1e-6 / self.dim
        }

        # Multi-start strategy
        for i in range(min(3, num_initial_samples)):  # Added multi-start loop
            starting_point = initial_solutions[i]
            result = minimize(wrapped_func, starting_point, method='L-BFGS-B', bounds=bounds, options=bfgs_options)
            
            if result.success and result.fun < best_score:
                best_score = result.fun
                best_solution = result.x

        return best_solution