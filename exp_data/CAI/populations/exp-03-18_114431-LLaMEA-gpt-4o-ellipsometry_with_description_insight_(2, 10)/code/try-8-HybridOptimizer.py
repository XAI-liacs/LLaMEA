import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def _initial_sample(self, lb, ub):
        return np.random.uniform(lb, ub, self.dim)

    def _objective_wrapper(self, func):
        def wrapped(x):
            if self.evaluations >= self.budget:
                raise Exception("Budget Exceeded")
            self.evaluations += 1
            return func(x)
        return wrapped
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Uniformly sample initial guesses
        initial_guess = self._initial_sample(lb, ub)
        
        # Adjusted bounds for optimization
        adjusted_bounds = [(max(l, i - 0.1 * (u - l)), min(u, i + 0.1 * (u - l)))
                           for l, u, i in zip(lb, ub, initial_guess)]
        
        # Optimize using BFGS with dynamic bounds adjustment
        result = minimize(self._objective_wrapper(func), initial_guess, method='L-BFGS-B',
                          bounds=adjusted_bounds, options={'maxiter': self.budget, 'ftol': 1e-9})

        return result.x if result.success else initial_guess