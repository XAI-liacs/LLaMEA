import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds from the function
        lb, ub = func.bounds.lb, func.bounds.ub

        # Ensure bounds are valid
        if np.any(ub <= lb):
            raise ValueError("All upper bounds must be greater than lower bounds.")

        # Calculate the number of initial samples dynamically based on remaining budget
        initial_sample_count = max(5, min(10, self.budget // 20))
        
        # Uniformly sample initial solutions within bounds
        initial_guesses = np.random.uniform(lb, ub, (initial_sample_count, self.dim))
        
        best_solution = None
        best_value = float('inf')
        evaluations = 0
        
        for guess in initial_guesses:
            # Use a local optimizer starting from each initial guess
            res = minimize(func, guess, method='L-BFGS-B', bounds=list(zip(lb, ub)), options={'maxfun': self.budget - evaluations})
            
            evaluations += res.nfev
            
            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x
            
            if evaluations >= self.budget:
                break

        return best_solution, best_value