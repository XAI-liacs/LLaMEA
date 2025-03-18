import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Calculate the number of initial samples using adaptive step size
        initial_samples = min(10, self.budget // 3)
        remaining_budget = self.budget - initial_samples
        
        # Extract bounds
        bounds = func.bounds
        lower_bounds, upper_bounds = np.array(bounds.lb), np.array(bounds.ub)
        
        # Adaptive uniform initial sampling based on parameter range
        step_size = (upper_bounds - lower_bounds) / (initial_samples + 1)
        initial_guesses = [lower_bounds + step_size * (i + 1) for i in range(initial_samples)]
        
        best_solution = None
        best_value = float('inf')
        
        # Evaluate initial solutions
        for guess in initial_guesses:
            val = func(guess)
            if val < best_value:
                best_value = val
                best_solution = guess
        
        # Local optimization using Trust-Region Reflective method
        def constrained_func(x):
            # Constraint function to keep within bounds
            return func(np.clip(x, lower_bounds, upper_bounds))
        
        # Modify bounds to be compatible with scipy's minimize function
        bounds_compatible = list(zip(lower_bounds, upper_bounds))
        
        result = minimize(constrained_func, best_solution, method='trust-constr', bounds=bounds_compatible, options={'maxiter': remaining_budget})
        
        return result.x