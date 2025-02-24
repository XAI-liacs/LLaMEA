import numpy as np
from scipy.optimize import minimize

class EnhancedAdaptiveNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize bounds and initial guesses
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        initial_guess = np.random.uniform(bounds[0], bounds[1], self.dim)
        
        # Define the optimization callback to count function evaluations
        self.func_evaluations = 0
        def callback(x):
            self.func_evaluations += 1
            if self.func_evaluations >= self.budget:
                raise StopIteration("Budget exhausted")

        # Perform optimization with dynamic boundary adjustments
        try:
            result = minimize(func, initial_guess, method='Nelder-Mead', 
                              bounds=bounds.T, callback=callback, 
                              options={'maxiter': self.budget, 'disp': False})
        except StopIteration:
            pass

        # Adjust bounds based on the found solution for potential further refinement
        center = result.x
        bounds = np.clip(bounds, func.bounds.lb, func.bounds.ub)
        shrink_factor = 0.2 if result.success else 0.1  # Shrink more if successful
        new_bounds = np.array([np.maximum(center - shrink_factor * (bounds[1] - bounds[0]), bounds[0]),
                               np.minimum(center + shrink_factor * (bounds[1] - bounds[0]), bounds[1])])

        # Restart optimization with updated bounds if budget allows
        while self.func_evaluations < self.budget:
            try:
                result = minimize(func, result.x, method='Nelder-Mead', 
                                  bounds=new_bounds.T, callback=callback, 
                                  options={'maxiter': min(self.budget - self.func_evaluations, 100), 'disp': False})
                # Further contraction on success
                if result.success:
                    new_bounds = np.array([np.maximum(result.x - 0.2 * (new_bounds[1] - new_bounds[0]), new_bounds[0]),
                                           np.minimum(result.x + 0.2 * (new_bounds[1] - new_bounds[0]), new_bounds[1])])
            except StopIteration:
                break

        return result.x