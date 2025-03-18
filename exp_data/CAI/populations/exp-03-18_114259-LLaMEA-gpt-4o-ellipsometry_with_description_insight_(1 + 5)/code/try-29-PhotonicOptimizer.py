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
        
        # Dynamically adjust number of initial samples based on remaining budget
        num_initial_samples = min(16 + self.budget // 100, self.budget // 10)  # Changed line
        
        # Generate initial samples by uniform sampling
        initial_samples = np.random.uniform(lb, ub, size=(num_initial_samples, self.dim))
        
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
            dynamic_radius = 0.05 * (ub - lb) * (1 + self.evaluations / self.budget)  # Changed line
            lb = np.maximum(lb, best_solution - dynamic_radius)  # Changed line
            ub = np.minimum(ub, best_solution + dynamic_radius)  # Changed line
            bounds = list(zip(lb, ub))
        
        return best_solution