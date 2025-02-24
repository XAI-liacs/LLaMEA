import numpy as np
from scipy.optimize import minimize

class ImprovedNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        scaling_factor = np.random.uniform(0.85, 1.15)
        initial_guess = np.random.uniform(bounds[0], bounds[1], self.dim) * scaling_factor
        
        self.func_evaluations = 0
        self.learning_rate = 0.1

        def callback(x):
            self.func_evaluations += 1
            if self.func_evaluations >= self.budget:
                raise StopIteration("Budget exhausted")
            if self.func_evaluations % 10 == 0:  # Adjust learning rate every 10 evaluations
                self.learning_rate *= 0.95
        
        try:
            dynamic_tol = 1e-6 * (1 + (self.func_evaluations / self.budget))
            result = minimize(func, initial_guess, method='Nelder-Mead', 
                              tol=dynamic_tol, callback=callback, 
                              options={'maxiter': self.budget, 'adaptive': True, 'disp': False})
            
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
                              options={'maxiter': self.budget - self.func_evaluations, 'adaptive': True, 'disp': False})
        except StopIteration:
            pass

        if self.func_evaluations < self.budget:
            weights = np.linspace(0.1, 0.9, self.dim)
            new_guess = np.random.uniform(bounds[0], bounds[1], self.dim) * weights
            try:
                result = minimize(func, new_guess, method='Nelder-Mead', 
                                  tol=1e-6, callback=callback, 
                                  options={'maxiter': self.budget - self.func_evaluations, 'adaptive': True, 'disp': False})
            except StopIteration:
                pass

        return result.x