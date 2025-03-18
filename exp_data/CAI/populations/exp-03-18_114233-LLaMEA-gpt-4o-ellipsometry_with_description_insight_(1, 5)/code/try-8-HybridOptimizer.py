import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        bounds = func.bounds
        lb = bounds.lb
        ub = bounds.ub
        
        num_samples = min(10, self.budget // 3)  # Adjust sampling strategy
        grids = [np.linspace(lb[i], ub[i], num_samples) for i in range(self.dim)]
        samples = np.array(np.meshgrid(*grids)).T.reshape(-1, self.dim)
        
        best_sample = None
        best_value = float('inf')
        evaluations = 0
        
        for sample in samples:
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_sample = sample
        
        # Adjust bounds dynamically based on best sample
        adaptive_bounds = [(max(lb[i], best_sample[i] - 0.1), min(ub[i], best_sample[i] + 0.1)) for i in range(self.dim)]
        
        # Use local optimization with dynamic bounds adjustment
        result = minimize(func, best_sample, bounds=adaptive_bounds, method='L-BFGS-B', options={'maxfun': self.budget - evaluations})
        return result.x