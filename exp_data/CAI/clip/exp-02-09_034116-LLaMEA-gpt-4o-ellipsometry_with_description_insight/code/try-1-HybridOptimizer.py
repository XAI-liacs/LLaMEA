import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')
        
        # Exploration phase: Uniform random sampling
        num_samples = max(1, self.budget // 10)  # Use 10% of budget for sampling
        for _ in range(num_samples):
            sample = np.random.uniform(lb, ub)
            value = func(sample)
            if value < best_value:
                best_value = value
                best_solution = sample
        
        # Exploitation phase: Local optimization using L-BFGS-B
        def wrapped_func(x):
            return func(x)
        
        result = minimize(wrapped_func, best_solution, method='L-BFGS-B', bounds=list(zip(lb, ub)), options={'maxiter': self.budget - num_samples})
        
        if result.fun < best_value:
            best_value = result.fun
            best_solution = result.x
        
        return best_solution