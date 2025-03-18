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
        step_size = 0.05 * (self.budget - self.func_evaluations) / self.budget
        new_bounds = np.array([np.maximum(center - step_size * (bounds[1] - bounds[0]), bounds[0]),
                               np.minimum(center + step_size * (bounds[1] - bounds[0]), bounds[1])])
        
        try:
            result = minimize(func, result.x, method='Nelder-Mead', 
                              tol=1e-6, callback=callback, 
                              options={'maxiter': self.budget - self.func_evaluations, 'disp': False})
        except StopIteration:
            pass

        if self.func_evaluations < self.budget:
            weights = np.linspace(0.1, 1.0, self.dim)
            new_guess = np.random.uniform(bounds[0], bounds[1], self.dim) * weights
            try:
                result = minimize(func, new_guess, method='Nelder-Mead', 
                                  tol=1e-6, callback=callback, 
                                  options={'maxiter': self.budget - self.func_evaluations, 'disp': False})
            except StopIteration:
                pass

        return result.x