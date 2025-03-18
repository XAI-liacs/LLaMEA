import numpy as np
from scipy.optimize import minimize

class AdaptiveNelderMead:
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
            
            # Early stopping if the solution has converged
            if result.success:
                return result.x
                
        except StopIteration:
            pass

        # Adjust bounds based on the found solution for potential further refinement
        center = result.x
        bounds = np.clip(bounds, func.bounds.lb, func.bounds.ub)
        new_bounds = np.array([np.maximum(center - 0.1 * (bounds[1] - bounds[0]), bounds[0]),
                               np.minimum(center + 0.1 * (bounds[1] - bounds[0]), bounds[1])])
        
        # Re-run optimization with updated bounds if budget allows
        try:
            result = minimize(func, initial_guess=result.x+0.05*(bounds[1]-bounds[0]), method='Nelder-Mead', 
                              bounds=new_bounds.T, callback=callback, 
                              options={'maxiter': self.budget - self.func_evaluations, 'disp': False})
        except StopIteration:
            pass

        return result.x