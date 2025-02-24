import numpy as np
from scipy.optimize import minimize

class HybridLocalGlobalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = [(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)]
        remaining_budget = self.budget
        
        # Step 1: Generate initial guesses using uniform random sampling
        num_initial_guesses = max(1, remaining_budget // self.dim)
        initial_guesses = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=(num_initial_guesses, self.dim))
        
        best_solution = None
        best_value = float('inf')
        
        # Step 2: Use local optimizer for each initial guess
        for initial_guess in initial_guesses:
            if remaining_budget <= 0:
                break
            
            # Adjust the options to include adaptive convergence tolerance
            result = minimize(func, initial_guess, method='L-BFGS-B', bounds=bounds, options={'maxfun': remaining_budget, 'ftol': 1e-6})
            remaining_budget -= result.nfev
            
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
        
        return best_solution

# Example usage:
# Assuming you have a function `func` with attributes `bounds.lb` and `bounds.ub`
# optimizer = HybridLocalGlobalOptimizer(budget=100, dim=2)
# best_solution = optimizer(func)