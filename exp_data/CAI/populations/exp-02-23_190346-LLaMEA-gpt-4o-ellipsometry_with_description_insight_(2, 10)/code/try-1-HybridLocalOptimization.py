import numpy as np
from scipy.optimize import minimize

class HybridLocalOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds
        lower_bounds, upper_bounds = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initial phase: Uniform sampling for initial guesses
        num_samples = min(10, self.budget // 2)
        samples = np.random.uniform(lower_bounds, upper_bounds, size=(num_samples, self.dim))
        best_sample = None
        best_value = float('inf')
        
        for sample in samples:
            value = func(sample)
            if value < best_value:
                best_value = value
                best_sample = sample
            self.budget -= 1
            if self.budget <= 0:
                return best_sample
        
        # Exploitation phase: Use Nelder-Mead for local optimization
        result = minimize(fun=func, x0=best_sample, method='Nelder-Mead', bounds=[(low, high) for low, high in zip(lower_bounds, upper_bounds)], options={'maxfev': self.budget})
        
        return result.x