import numpy as np
from scipy.optimize import minimize

class AdaptiveNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        scaling_factor = np.random.uniform(0.8, 1.2)  # Adjusted dynamic scaling factor range
        initial_guess = np.random.uniform(bounds[0], bounds[1], self.dim) * scaling_factor
        
        self.func_evaluations = 0
        def callback(x):
            self.func_evaluations += 1
            if self.func_evaluations >= self.budget:
                raise StopIteration("Budget exhausted")

        try:
            dynamic_tol = 1e-6 * (1 + np.exp(self.func_evaluations / self.budget))  # Modified dynamic tolerance calculation
            result = minimize(func, initial_guess, method='Nelder-Mead', 
                              tol=dynamic_tol, callback=callback, 
                              options={'maxiter': self.budget, 'disp': False})
            
            if result.success:
                return result.x
                
        except StopIteration:
            pass

        center = result.x
        local_sampling_factor = np.random.uniform(0.15, 0.25)  # Modified adaptive local sampling factor
        new_bounds = np.array([np.maximum(center - local_sampling_factor * (bounds[1] - bounds[0]), bounds[0]),
                               np.minimum(center + local_sampling_factor * (bounds[1] - bounds[0]), bounds[1])])
        
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