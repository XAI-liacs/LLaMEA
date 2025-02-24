import numpy as np
from scipy.optimize import minimize

class HybridOpt:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        # Extract bounds
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialize best solution
        best_x = None
        best_fval = np.inf
        
        # Calculate the number of initial guesses based on budget
        initial_guesses = min(self.budget // 10, 10)
        
        # Use uniform sampling for initial guesses
        for _ in range(initial_guesses):
            x0 = lb + (ub - lb) * np.random.rand(self.dim)
            
            # Local optimization using Nelder-Mead
            result = minimize(func, x0, method='Nelder-Mead', bounds=list(zip(lb, ub)))
            
            # Update best solution found
            if result.fun < best_fval:
                best_fval = result.fun
                best_x = result.x
                
            # Decrement budget
            self.budget -= result.nfev
            if self.budget <= 0:
                break
        
        return best_x, best_fval