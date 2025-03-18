import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds from the function
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Number of initial samples
        num_samples = min(10, self.budget // 2)
        
        # Uniform sampling to initialize
        samples = np.random.uniform(lb, ub, (num_samples, self.dim))
        sample_vals = [func(sample) for sample in samples]
        
        # Check budget usage
        self.budget -= num_samples
        
        # Find the best initial solution
        best_idx = np.argmin(sample_vals)
        best_sample = samples[best_idx]
        
        # Define a function wrapper to track budget
        def budgeted_func(x):
            if self.budget <= 0:
                raise RuntimeError("Budget exceeded")
            self.budget -= 1
            return func(x)
        
        # Use trust-region for local optimization
        result = minimize(budgeted_func, best_sample, method='trust-constr', 
                          bounds=list(zip(lb, ub)))
        
        return result.x if result.success else best_sample