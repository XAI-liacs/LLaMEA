import numpy as np
from scipy.optimize import minimize

class AdvancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        initial_samples = max(min(self.budget // (3 * self.dim), 100), 10)
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
        
        # Hybrid BFGS and Nelder-Mead strategy based on remaining budget
        if remaining_budget > 0.2 * self.budget:  # Use L-BFGS-B if sufficient budget remains
            options = {'maxiter': remaining_budget, 'disp': False}
            result = minimize(bounded_func, best_solution, method='L-BFGS-B', bounds=np.array([lb, ub]).T, options=options)
        else:  # Switch to Nelder-Mead for exploitation under tight budget
            options = {'maxiter': remaining_budget, 'disp': False}
            result = minimize(bounded_func, best_solution, method='Nelder-Mead', options=options)
        
        return result.x