import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Halton

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Define the bounds based on the function's bounds
        bounds = func.bounds
        lb = bounds.lb
        ub = bounds.ub
        
        # Modify: Use Halton sequence for quasi-random sampling
        sampler = Halton(d=self.dim, scramble=True)
        num_samples = min(10, self.budget // 2)
        samples = lb + (ub - lb) * sampler.random(num_samples)
        
        # Evaluate initial samples and store the best
        best_sample = None
        best_value = float('inf')
        evaluations = 0
        
        for sample in samples:
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_sample = sample
        
        # Use BFGS for local optimization starting from the best sample found
        result = minimize(func, best_sample, bounds=list(zip(lb, ub)), method='L-BFGS-B', options={'maxfun': self.budget - evaluations})
        return result.x