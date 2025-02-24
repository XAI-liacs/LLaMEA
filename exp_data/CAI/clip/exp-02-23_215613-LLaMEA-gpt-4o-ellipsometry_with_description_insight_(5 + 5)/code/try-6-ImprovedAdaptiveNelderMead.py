import numpy as np
from scipy.optimize import minimize

class ImprovedAdaptiveNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        initial_guess = np.random.uniform(bounds[:, 0], bounds[:, 1])
        result = self.adaptive_nelder_mead(func, initial_guess, bounds)
        return result.x

    def adaptive_nelder_mead(self, func, initial_guess, bounds):
        options = {'maxiter': self.budget - self.evaluations, 'adaptive': True}
        
        def bounded_func(x):
            self.evaluations += 1
            if np.all(x >= bounds[:, 0]) and np.all(x <= bounds[:, 1]):
                return func(x)
            else:
                return float('inf')  # Penalize out-of-bound solutions

        result = minimize(bounded_func, initial_guess, method='Nelder-Mead', options=options)
        
        # Dynamically refine bounds based on the result
        if self.evaluations < self.budget:
            margin = (bounds[:, 1] - bounds[:, 0]) * 0.1
            new_lb = np.maximum(bounds[:, 0], result.x - margin)
            new_ub = np.minimum(bounds[:, 1], result.x + margin)
            new_bounds = np.column_stack((new_lb, new_ub))
            initial_guess = np.random.uniform(new_bounds[:, 0], new_bounds[:, 1])
            return self.adaptive_nelder_mead(func, initial_guess, new_bounds)
        
        return result