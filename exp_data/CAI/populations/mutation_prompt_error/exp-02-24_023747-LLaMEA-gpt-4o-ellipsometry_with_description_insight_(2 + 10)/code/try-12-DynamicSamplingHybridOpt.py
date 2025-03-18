import numpy as np
from scipy.optimize import minimize

class DynamicSamplingHybridOpt:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        # Extract bounds
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialize best solution
        best_x = None
        best_fval = np.inf
        
        # Calculate the number of initial guesses based on remaining budget
        initial_guesses = max(1, self.budget // 10)  # Increase number of initial guesses
        
        def adjust_bounds(center, scale=0.1):
            """ Adjust bounds around a center point to focus search """
            return np.clip(center + scale * (np.random.rand(self.dim) - 0.5) * (ub - lb), lb, ub)
        
        for _ in range(initial_guesses):
            # Dynamic sampling around initial guess
            x0 = lb + (ub - lb) * np.random.rand(self.dim)
            
            # Refine initial guess using local bounds adjustment
            local_lb = adjust_bounds(x0, scale=0.05)
            local_ub = adjust_bounds(x0, scale=0.05)
            
            # Local optimization using Nelder-Mead with adjusted bounds
            result = minimize(func, x0, method='Nelder-Mead', bounds=list(zip(local_lb, local_ub)))

            # Update best solution found
            if result.fun < best_fval:
                best_fval = result.fun
                best_x = result.x
            
            # Decrement budget
            self.budget -= result.nfev
            if self.budget <= 0:
                break

        return best_x, best_fval