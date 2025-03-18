import numpy as np
from scipy.optimize import minimize

class AdvancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Tiered initial sampling for better exploration
        initial_samples = max(min(self.budget // (3 * self.dim), 100), 15)
        remaining_budget = self.budget - initial_samples
        
        # Uniformly sample initial points with refined dynamic strategy
        samples = np.random.uniform(lb, ub, (initial_samples, self.dim))
        best_value = float('inf')
        best_solution = None
        
        # Evaluate sampled points
        evaluations = 0
        for sample in samples:
            if evaluations >= self.budget:
                break
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample
        
        # Define a bounded function to ensure the search remains within the specified bounds
        def bounded_func(x):
            return func(np.clip(x, lb, ub))
        
        # Use dual local optimizers: L-BFGS-B and a fallback to Nelder-Mead if needed
        options = {'maxiter': remaining_budget // 2, 'disp': False}
        result = minimize(bounded_func, best_solution, method='L-BFGS-B', bounds=np.array([lb, ub]).T, options=options)
        
        if not result.success and remaining_budget > 0:
            options = {'maxiter': remaining_budget, 'disp': False}
            result = minimize(bounded_func, best_solution, method='Nelder-Mead', options=options)
        
        return result.x