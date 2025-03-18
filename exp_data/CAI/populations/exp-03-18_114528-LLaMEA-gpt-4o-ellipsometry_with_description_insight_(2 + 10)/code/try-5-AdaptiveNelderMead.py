import numpy as np
from scipy.optimize import minimize

class AdaptiveNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initial uniform sampling within bounds
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        initial_point = np.random.uniform(bounds[0], bounds[1], size=self.dim)

        # Initialize budget counter
        evaluations = 0

        def callback(xk):
            nonlocal evaluations
            # Dynamic bound tightening based on current best solution
            current_bounds = np.clip(bounds, xk - 0.1, xk + 0.1)
            bounds[0] = np.maximum(bounds[0], current_bounds[0])
            bounds[1] = np.minimum(bounds[1], current_bounds[1])
            # Reallocate budget based on progress
            if evaluations < self.budget * 0.5:
                self.budget += int(self.budget * 0.1)

        # Optimization with Nelder-Mead method
        result = minimize(
            func,
            initial_point,
            method='Nelder-Mead',
            callback=callback,
            options={'maxiter': self.budget, 'adaptive': True}
        )
        
        return result.x, result.fun