import numpy as np
from scipy.optimize import minimize

class AdaptiveBoundaryRefinement:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Initialize bounds
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Uniformly sample the initial points with increased density
        initial_guesses = np.random.uniform(lb, ub, (5 * self.dim, len(lb)))  # Changed from 2 to 5
        
        best_solution = None
        best_value = float('inf')
        evaluations = 0
        
        for guess in initial_guesses:
            if evaluations >= self.budget:
                break
            
            # Local optimization using Nelder-Mead
            result = minimize(func, guess, method='Nelder-Mead', bounds=list(zip(lb, ub)))
            evaluations += result.nfev
            
            # Check if we found a new best solution
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            
            # Update bounds based on the best solution found so far
            if evaluations < self.budget:
                margin = np.clip(0.02 * (ub - lb), 1e-5, None)  # Changed from 0.05 to 0.02
                lb = np.maximum(func.bounds.lb, best_solution - margin)
                ub = np.minimum(func.bounds.ub, best_solution + margin)
        
        return best_solution