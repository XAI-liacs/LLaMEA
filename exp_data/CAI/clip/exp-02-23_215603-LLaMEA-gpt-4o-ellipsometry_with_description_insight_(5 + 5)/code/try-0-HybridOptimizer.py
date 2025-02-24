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
        
        # Uniform sampling phase
        num_samples = min(10, self.budget // 2)
        samples = np.random.uniform(func.bounds.lb, func.bounds.ub, (num_samples, self.dim))
        
        for sample in samples:
            value = func(sample)
            self.evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample
            if self.evaluations >= self.budget:
                return best_solution
        
        # Local optimization with BFGS
        def wrapped_func(x):
            self.evaluations += 1
            if self.evaluations > self.budget:
                raise RuntimeError("Budget exceeded")
            return func(x)
        
        bounds = [(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)]
        result = minimize(wrapped_func, best_solution, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - self.evaluations})
        
        if result.success and result.fun < best_value:
            best_solution = result.x
        
        return best_solution