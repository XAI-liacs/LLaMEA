import numpy as np
from scipy.optimize import minimize

class AdaptiveNelderMeadDynamicControl:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        self.func_evaluations = 0
        initial_guess = np.random.uniform(bounds[0], bounds[1], self.dim)
        
        def callback(x):
            self.func_evaluations += 1
            if self.func_evaluations >= self.budget:
                raise StopIteration("Budget exhausted")

        try:
            dynamic_tol = 1e-6 * (1 + (self.func_evaluations / self.budget)**2)
            result = minimize(func, initial_guess, method='Nelder-Mead', 
                              tol=dynamic_tol, callback=callback, 
                              options={'maxiter': self.budget, 'disp': False})
            
            if result.success:
                center = result.x

        except StopIteration:
            return result.x

        while self.func_evaluations < self.budget:
            exploration_phase = self.func_evaluations < self.budget // 2
            if exploration_phase:
                scaling_factor = np.random.uniform(0.9, 1.1)
                new_guess = np.random.uniform(bounds[0], bounds[1], self.dim) * scaling_factor
            else:
                adjustment_factor = max(0.1, 1 - (self.func_evaluations / self.budget))
                new_bounds = np.array([np.maximum(center - adjustment_factor * (bounds[1] - bounds[0]), bounds[0]),
                                       np.minimum(center + adjustment_factor * (bounds[1] - bounds[0]), bounds[1])])
                new_guess = np.random.uniform(new_bounds[0], new_bounds[1], self.dim)

            try:
                result = minimize(func, new_guess, method='Nelder-Mead', 
                                  tol=1e-6, callback=callback, 
                                  options={'maxiter': self.budget - self.func_evaluations, 'disp': False})
                
                if result.success:
                    center = result.x

            except StopIteration:
                break

        return result.x