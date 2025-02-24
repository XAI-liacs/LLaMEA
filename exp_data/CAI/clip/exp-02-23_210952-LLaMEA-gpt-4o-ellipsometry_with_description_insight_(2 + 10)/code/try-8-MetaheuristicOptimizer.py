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

        num_initial_samples = min(self.budget // 3, 15)  # Increased exploration proportion
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

        # Increase precision of the local optimizer
        result = minimize(wrapped_func, best_solution, method='L-BFGS-B', bounds=bounds, options={'maxfun': remaining_budget, 'ftol': 1e-9})

        return result.x if result.success else best_solution