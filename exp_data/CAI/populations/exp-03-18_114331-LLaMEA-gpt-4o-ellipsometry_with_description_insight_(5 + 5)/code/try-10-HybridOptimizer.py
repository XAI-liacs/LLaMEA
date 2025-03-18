import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        num_initial_samples = int(self.budget * 0.2)  # Changed from 0.1 to 0.2
        remaining_budget = self.budget - num_initial_samples
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        
        # Uniformly sample initial points within bounds
        samples = np.random.uniform(func.bounds.lb, func.bounds.ub, (num_initial_samples, self.dim))
        best_sample = None
        best_sample_value = float('inf')
        
        # Evaluate sampled points
        for sample in samples:
            value = func(sample)
            if value < best_sample_value:
                best_sample_value = value
                best_sample = sample
        
        # Use BFGS local optimization from the best initial sample
        def wrapped_func(x):
            nonlocal remaining_budget
            if remaining_budget <= 0:
                return float('inf')
            remaining_budget -= 1
            return func(x)
        
        result = minimize(wrapped_func, best_sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': remaining_budget})
        
        return result.x