import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Step 1: Initial global exploration with uniform sampling
        initial_samples = int(self.budget * 0.2)  # 20% of budget for initial exploration
        best_sample = None
        best_value = float('inf')
        
        for _ in range(initial_samples):
            x0 = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
            value = func(x0)
            if value < best_value:
                best_value = value
                best_sample = x0
        
        # Step 2: Local optimization using BFGS
        remaining_budget = self.budget - initial_samples
        if remaining_budget > 0:
            result = minimize(func, best_sample, method='BFGS', 
                              bounds=np.array([func.bounds.lb, func.bounds.ub]).T,
                              options={'maxfun': remaining_budget})
            best_sample = result.x
            best_value = result.fun
        
        return best_sample, best_value