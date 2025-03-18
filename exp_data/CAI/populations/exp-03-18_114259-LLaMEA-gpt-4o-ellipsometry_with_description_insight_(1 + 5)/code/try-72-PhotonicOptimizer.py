import numpy as np
from scipy.optimize import minimize

class PhotonicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
    
    def __call__(self, func):
        # Get bounds from func
        lb = func.bounds.lb
        ub = func.bounds.ub
        bounds = list(zip(lb, ub))
        
        # Number of initial samples for uniform sampling
        num_initial_samples = min(20, self.budget // 10)  # Changed from 16 to 20
        
        # Generate initial samples by Gaussian sampling (changed from uniform)
        initial_samples = np.random.normal(loc=(lb + ub) / 2, scale=(ub - lb) / 4, size=(num_initial_samples, self.dim))
        initial_samples = np.clip(initial_samples, lb, ub)

        best_solution = None
        best_value = float('inf')
        
        for i, sample in enumerate(initial_samples):
            if self.evaluations >= self.budget:
                break
            
            # Local optimization using L-BFGS-B starting from each sample
            res = minimize(func, sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - self.evaluations})
            self.evaluations += res.nfev
            
            # Update best solution found so far
            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x
            
            # Iteratively adjust bounds based on the best solution found
            radius = 0.05 * (ub - lb)  # Changed from 0.1 to 0.05
            lb = np.maximum(lb, best_solution - radius)
            ub = np.minimum(ub, best_solution + radius)
            bounds = list(zip(lb, ub))
        
        return best_solution