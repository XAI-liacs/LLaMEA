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
        
        num_initial_samples = min(10, self.budget // 10)
        
        initial_samples = np.random.uniform(lb, ub, size=(num_initial_samples, self.dim))
        
        best_solution = None
        best_value = float('inf')
        
        for i, sample in enumerate(initial_samples):
            if self.evaluations >= self.budget:
                break
            
            res = minimize(func, sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - self.evaluations})
            self.evaluations += res.nfev
            
            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x
            
            radius = 0.1 * (ub - lb) * (0.9 ** (i / num_initial_samples))  # Introducing annealing
            lb = np.maximum(func.bounds.lb, best_solution - radius)  # Ensuring bounds do not shrink beyond initial
            ub = np.minimum(func.bounds.ub, best_solution + radius)  # bounds
            bounds = list(zip(lb, ub))
        
        return best_solution