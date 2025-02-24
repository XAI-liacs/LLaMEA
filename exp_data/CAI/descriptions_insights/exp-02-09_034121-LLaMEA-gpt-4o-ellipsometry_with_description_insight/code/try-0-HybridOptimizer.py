import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initial uniform sampling to ensure good coverage
        samples = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.budget // 2, self.dim))
        evaluations = [func(sample) for sample in samples]
        self.budget -= len(samples)
        
        # Select the best initial sample
        best_sample_idx = np.argmin(evaluations)
        best_sample = samples[best_sample_idx]
        best_value = evaluations[best_sample_idx]
        
        # Bounds for the optimizer
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        
        # BFGS optimization from the best initial sample
        def wrapped_func(x):
            nonlocal best_value
            if self.budget <= 0:
                return best_value + 1e6  # Large penalty to stop further evaluation
            result = func(x)
            self.budget -= 1
            if result < best_value:
                best_value = result
            return result
        
        result = minimize(wrapped_func, best_sample, method='L-BFGS-B', bounds=bounds)
        
        return result.x if result.success else best_sample