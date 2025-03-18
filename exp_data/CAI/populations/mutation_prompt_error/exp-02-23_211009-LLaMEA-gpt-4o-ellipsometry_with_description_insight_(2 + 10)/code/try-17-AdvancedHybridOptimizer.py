import numpy as np
from scipy.optimize import minimize

class AdvancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        initial_samples = max(min(self.budget // (2 * self.dim), 100), 10)
        remaining_budget = self.budget - initial_samples
        
        samples = np.random.uniform(lb, ub, (initial_samples, self.dim))
        best_value = float('inf')
        best_solution = None
        
        evaluations = 0
        for sample in samples:
            if evaluations >= self.budget:
                break
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample
        
        def bounded_func(x):
            return func(np.clip(x, lb, ub))
        
        # Change: Use different local search strategy based on remaining budget
        method = 'BFGS' if remaining_budget > 50 else 'Nelder-Mead'
        options = {'maxiter': remaining_budget, 'disp': False}
        result = minimize(bounded_func, best_solution, method=method, options=options)
        
        return result.x