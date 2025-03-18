import numpy as np
from scipy.optimize import minimize

class PhotonicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
    
    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        bounds = list(zip(lb, ub))
        
        # Dynamically adjust initial samples based on budget
        num_initial_samples = min(16, self.budget // 8)  # Changed divisor from 10 to 8
        
        initial_samples = np.random.uniform(lb, ub, size=(num_initial_samples, self.dim))
        
        best_solution = None
        best_value = float('inf')
        
        for i, sample in enumerate(initial_samples):
            if self.evaluations >= self.budget:
                break
            
            # Switch method based on remaining budget
            method = 'L-BFGS-B' if self.budget - self.evaluations > 50 else 'Nelder-Mead'  # Changed method selection based on budget
            res = minimize(func, sample, method=method, bounds=bounds if method == 'L-BFGS-B' else None, options={'maxfun': self.budget - self.evaluations})
            self.evaluations += res.nfev
            
            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x
            
            radius = 0.05 * (ub - lb)
            lb = np.maximum(lb, best_solution - radius)
            ub = np.minimum(ub, best_solution + radius)
            bounds = list(zip(lb, ub))
        
        return best_solution