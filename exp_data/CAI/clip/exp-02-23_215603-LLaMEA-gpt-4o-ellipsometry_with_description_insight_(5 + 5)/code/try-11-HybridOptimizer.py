import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
    
    def __call__(self, func):
        # Initialize best solution and its function value
        best_solution = None
        best_value = float('inf')
        
        # Adaptive uniform sampling phase
        adaptive_samples = max(5, self.budget // 3)
        samples = np.random.uniform(func.bounds.lb, func.bounds.ub, (adaptive_samples, self.dim))
        
        for sample in samples:
            value = func(sample)
            self.evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample
            if self.evaluations >= self.budget:
                return best_solution
        
        # Local optimization with Trust-Region-like adjustment
        def wrapped_func(x):
            self.evaluations += 1
            if self.evaluations > self.budget:
                raise RuntimeError("Budget exceeded")
            return func(x)
        
        trust_radius = 0.1 * (func.bounds.ub - func.bounds.lb)
        options = {'maxfun': self.budget - self.evaluations, 'eps': trust_radius}
        bounds = [(max(lb, best_solution[i] - trust_radius[i]), 
                   min(ub, best_solution[i] + trust_radius[i])) for i, (lb, ub) in enumerate(zip(func.bounds.lb, func.bounds.ub))]
        
        result = minimize(wrapped_func, best_solution, method='L-BFGS-B', bounds=bounds, options=options)
        
        if result.success and result.fun < best_value:
            best_solution = result.x
        
        return best_solution