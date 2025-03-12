import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

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
        
        # Calculate the number of initial guesses based on remaining budget
        initial_guesses = max(1, self.budget // 15)  # Adjusted line
        
        # Use Sobol sequence for initial guesses
        sampler = Sobol(d=self.dim, scramble=True)
        sample_points = sampler.random_base2(m=int(np.ceil(np.log2(initial_guesses))))
        
        for i in range(initial_guesses):
            x0 = lb + (ub - lb) * sample_points[i]
            
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