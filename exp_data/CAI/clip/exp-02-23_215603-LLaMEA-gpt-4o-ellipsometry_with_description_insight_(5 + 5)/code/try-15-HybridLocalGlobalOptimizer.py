import numpy as np
from scipy.optimize import minimize

class HybridLocalGlobalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = [(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)]
        remaining_budget = self.budget
        
        # Step 1: Generate initial guesses using Gaussian sampling centered at midpoint
        num_initial_guesses = min(max(2, remaining_budget // 4), 10)
        midpoint = (func.bounds.lb + func.bounds.ub) / 2
        initial_guesses = np.clip(np.random.normal(loc=midpoint, scale=0.1, size=(num_initial_guesses, self.dim)), func.bounds.lb, func.bounds.ub)
        
        best_solution = None
        best_value = float('inf')
        
        # Step 2: Use local optimizer for each initial guess
        for initial_guess in initial_guesses:
            if remaining_budget <= 0:
                break
            
            result = minimize(func, initial_guess, method='L-BFGS-B', bounds=bounds, options={'maxiter': remaining_budget})
            remaining_budget -= result.nfev
            
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
        
        return best_solution