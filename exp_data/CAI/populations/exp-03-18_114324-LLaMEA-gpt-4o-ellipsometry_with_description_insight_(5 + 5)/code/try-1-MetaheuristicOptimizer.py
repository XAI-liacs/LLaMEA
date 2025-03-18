import numpy as np
from scipy.optimize import minimize

class MetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        # Initialize with uniform random sampling
        num_initial_samples = min(5, self.budget // 2)  # Limit initial samples
        initial_samples = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_initial_samples, self.dim))
        
        best_value = float('inf')
        best_solution = None

        # Evaluate initial samples
        for sample in initial_samples:
            value = func(sample)
            if value < best_value:
                best_value = value
                best_solution = sample

        remaining_budget = self.budget - num_initial_samples

        # Optimize using Nelder-Mead method
        result = minimize(
            func, 
            x0=best_solution, 
            method='Nelder-Mead', 
            bounds=bounds,
            options={'maxfev': remaining_budget, 'disp': False}
        )

        return result.x, result.fun