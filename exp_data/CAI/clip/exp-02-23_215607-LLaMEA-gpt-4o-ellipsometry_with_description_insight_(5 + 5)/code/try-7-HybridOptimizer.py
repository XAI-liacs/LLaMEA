import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Calculate the number of initial samples
        initial_samples = min(10, self.budget // 2)
        remaining_budget = self.budget - initial_samples
        
        # Extract bounds
        bounds = func.bounds
        lower_bounds, upper_bounds = np.array(bounds.lb), np.array(bounds.ub)
        
        # Uniform initial sampling
        initial_guesses = np.random.uniform(lower_bounds, upper_bounds, (initial_samples, self.dim))
        best_solution = None
        best_value = float('inf')
        
        # Evaluate initial solutions
        for guess in initial_guesses:
            val = func(guess)
            if val < best_value:
                best_value = val
                best_solution = guess
        
        # Local optimization using BFGS
        def constrained_func(x):
            # Constraint function to keep within bounds
            return func(np.clip(x, lower_bounds, upper_bounds))
        
        # Modify bounds to be compatible with scipy's minimize function
        bounds_compatible = list(zip(lower_bounds, upper_bounds))
        
        result = minimize(constrained_func, best_solution, method='L-BFGS-B', bounds=bounds_compatible, options={'maxfun': remaining_budget})
        
        return result.x