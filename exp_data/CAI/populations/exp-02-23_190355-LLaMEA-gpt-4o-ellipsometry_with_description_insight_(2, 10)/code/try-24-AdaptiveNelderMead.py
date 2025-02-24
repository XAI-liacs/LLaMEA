import numpy as np
from scipy.optimize import minimize

class AdaptiveNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        initial_guess = np.random.uniform(bounds[0], bounds[1], self.dim)
        
        self.func_evaluations = 0
        def callback(x):
            self.func_evaluations += 1
            if self.func_evaluations >= self.budget:
                raise StopIteration("Budget exhausted")

        try:
            result = minimize(func, initial_guess, method='Nelder-Mead', 
                              tol=1e-6, callback=callback, 
                              options={'maxiter': self.budget, 'disp': False})
            
            if result.success:
                return result.x
                
        except StopIteration:
            pass

        center = result.x
        bounds = np.clip(bounds, func.bounds.lb, func.bounds.ub)
        new_bounds = np.array([np.maximum(center - 0.1 * (bounds[1] - bounds[0]), bounds[0]),
                               np.minimum(center + 0.1 * (bounds[1] - bounds[0]), bounds[1])])
        
        try:
            result = minimize(func, result.x, method='Nelder-Mead', 
                              tol=1e-6, callback=callback, 
                              options={'maxiter': self.budget - self.func_evaluations, 'disp': False})
        except StopIteration:
            pass

        if self.func_evaluations < self.budget:
            weights = np.linspace(0.5, 1.5, self.dim)  # Adjusted dynamic weighting range
            new_guess = np.random.uniform(new_bounds[0], new_bounds[1], self.dim) * weights
            try:
                result = minimize(func, new_guess, method='Nelder-Mead', 
                                  tol=1e-6, callback=callback, 
                                  options={'maxiter': self.budget - self.func_evaluations, 'disp': False})
            except StopIteration:
                pass

        return result.x