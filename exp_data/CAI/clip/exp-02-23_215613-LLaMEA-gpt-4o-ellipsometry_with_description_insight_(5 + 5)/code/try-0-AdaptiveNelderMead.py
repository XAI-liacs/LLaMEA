import numpy as np
from scipy.optimize import minimize

class AdaptiveNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        bounds = func.bounds
        initial_guess = np.random.uniform(bounds.lb, bounds.ub)
        result = self.adaptive_nelder_mead(func, initial_guess, bounds)
        return result.x

    def adaptive_nelder_mead(self, func, initial_guess, bounds):
        options = {'maxiter': self.budget, 'adaptive': True}
        
        def bounded_func(x):
            self.evaluations += 1
            if np.all(x >= bounds.lb) and np.all(x <= bounds.ub):
                return func(x)
            else:
                return float('inf')  # Penalize out-of-bound solutions

        result = minimize(bounded_func, initial_guess, method='Nelder-Mead', options=options)
        
        # Dynamically adjust bounds based on the result
        if self.evaluations < self.budget:
            margin = (bounds.ub - bounds.lb) * 0.1
            new_lb = np.maximum(bounds.lb, result.x - margin)
            new_ub = np.minimum(bounds.ub, result.x + margin)
            new_bounds = np.array([new_lb, new_ub])
            initial_guess = np.random.uniform(new_bounds[0], new_bounds[1])
            return self.adaptive_nelder_mead(func, initial_guess, new_bounds)
        
        return result