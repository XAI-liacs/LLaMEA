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
        initial_guesses = np.random.uniform(lb, ub, (2 * self.dim, len(lb)))
        
        best_solution = None
        best_value = float('inf')
        evaluations = 0
        
        for guess in initial_guesses:
            if evaluations >= self.budget:
                break
            
            # Local optimization using hybrid approach
            result = minimize(func, guess, method='L-BFGS-B', bounds=list(zip(lb, ub)))
            evaluations += result.nfev
            
            # Check if we found a new best solution
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            
            # Update bounds based on the best solution found so far
            if evaluations < self.budget:
                margin = np.clip(0.03 * (ub - lb) * (result.fun / best_value), 1e-5, None)  # 3% margin, dynamic adjustment
                lb = np.maximum(func.bounds.lb, best_solution - margin)
                ub = np.minimum(func.bounds.ub, best_solution + margin)
        
        return best_solution