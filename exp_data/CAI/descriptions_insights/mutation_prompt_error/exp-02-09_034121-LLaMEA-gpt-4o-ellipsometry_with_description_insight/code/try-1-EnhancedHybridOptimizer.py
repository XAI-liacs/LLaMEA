import numpy as np
from scipy.optimize import minimize

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initial Gaussian sampling to refine initial guesses
        samples = np.random.normal((func.bounds.lb + func.bounds.ub) / 2, 
                                   (func.bounds.ub - func.bounds.lb) / 6, 
                                   (self.budget // 2, self.dim))
        samples = np.clip(samples, func.bounds.lb, func.bounds.ub)
        evaluations = [func(sample) for sample in samples]
        self.budget -= len(samples)
        
        # Select the best initial sample
        best_sample_idx = np.argmin(evaluations)
        best_sample = samples[best_sample_idx]
        best_value = evaluations[best_sample_idx]
        
        # Adaptive bounds based on the best sample and previous samples
        adaptive_bounds = [(max(func.bounds.lb[i], min(samples[:, i])), 
                            min(func.bounds.ub[i], max(samples[:, i]))) for i in range(self.dim)]
        
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
        
        result = minimize(wrapped_func, best_sample, method='L-BFGS-B', bounds=adaptive_bounds)
        
        return result.x if result.success else best_sample