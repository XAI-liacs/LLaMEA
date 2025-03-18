import numpy as np
from scipy.optimize import minimize

class AdaptiveNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        # Start with a uniform sampling to get initial guess
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        initial_guess = np.mean(bounds, axis=1)
        
        # Define the callback function to track evaluation budget
        def callback(xk):
            self.eval_count += 1
            if self.eval_count >= self.budget:
                raise StopIteration("Budget exhausted")
        
        # Define a function to wrap the original function with budget control
        def func_with_budget(x):
            if self.eval_count >= self.budget:
                raise StopIteration("Budget exhausted")
            self.eval_count += 1
            return func(x)
        
        self.eval_count = 0
        
        try:
            # Use Nelder-Mead with adaptive bounds
            result = minimize(
                func_with_budget, 
                initial_guess, 
                method='Nelder-Mead', 
                bounds=bounds,
                callback=callback,
                options={'maxiter': self.budget, 'adaptive': True}
            )
        except StopIteration:
            # Catch StopIteration to handle budget limit
            pass
        
        # Refine bounds and iteratively improve
        if self.eval_count < self.budget:
            refined_bounds = [(max(func.bounds.lb[i], result.x[i] - 0.1 * (func.bounds.ub[i] - func.bounds.lb[i])), 
                               min(func.bounds.ub[i], result.x[i] + 0.1 * (func.bounds.ub[i] - func.bounds.lb[i])))
                              for i in range(self.dim)]
            try:
                result = minimize(
                    func_with_budget, 
                    result.x, 
                    method='Nelder-Mead', 
                    bounds=refined_bounds,
                    callback=callback,
                    options={'maxiter': self.budget - self.eval_count, 'adaptive': True}
                )
            except StopIteration:
                pass
        
        return result.x