import numpy as np
from scipy.optimize import minimize

class DynamicMarginLocalExplorer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Initialize bounds
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Increase initial sampling density for better coverage
        initial_guesses = np.random.uniform(lb, ub, (3 * self.dim, len(lb)))
        
        best_solution = None
        best_value = float('inf')
        evaluations = 0
        
        for guess in initial_guesses:
            if evaluations >= self.budget:
                break
            
            # Local optimization using Nelder-Mead with refined bounds
            result = minimize(func, guess, method='Nelder-Mead', bounds=list(zip(lb, ub)))
            evaluations += result.nfev
            
            # Check for a new best solution
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            
            # Dynamically adjust bounds based on the best solution found
            if evaluations < self.budget:
                # Calculate dynamic margin based on current best solution
                margin_factor = 0.1 * np.sin(np.pi * evaluations / self.budget)  # Sine-based dynamic scaling
                margin = np.clip(margin_factor * (ub - lb), 1e-5, None)
                
                # Update bounds to focus search
                lb = np.maximum(func.bounds.lb, best_solution - margin)
                ub = np.minimum(func.bounds.ub, best_solution + margin)
        
        return best_solution