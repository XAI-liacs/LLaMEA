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
        
        num_initial_samples = min(18, self.budget // 8)  # Changed from 16 to 18, changed budget division
        
        initial_samples = np.random.uniform(lb, ub, size=(num_initial_samples, self.dim))
        
        best_solution = None
        best_value = float('inf')
        
        for i, sample in enumerate(initial_samples):
            if self.evaluations >= self.budget:
                break
            
            gradient = np.random.uniform(-0.1, 0.1, self.dim)  # New gradient perturbation
            sample = np.clip(sample + gradient, lb, ub)  # New direct application of gradient
            
            res = minimize(func, sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - self.evaluations})
            self.evaluations += res.nfev
            
            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x
            
            radius = 0.03 * (ub - lb)  # Changed from 0.05 to 0.03
            lb = np.maximum(lb, best_solution - radius)
            ub = np.minimum(ub, best_solution + radius)
            bounds = list(zip(lb, ub))
        
        return best_solution