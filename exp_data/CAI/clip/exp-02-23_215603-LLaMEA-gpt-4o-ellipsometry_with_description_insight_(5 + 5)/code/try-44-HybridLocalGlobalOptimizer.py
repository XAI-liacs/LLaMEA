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
        num_initial_guesses = min(max(2, remaining_budget // 4), 10)
        initial_guesses = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=(num_initial_guesses, self.dim))
        
        best_solution = None
        best_value = float('inf')
        
        # Step 2: Use local optimizer for each initial guess
        for initial_guess in initial_guesses:
            if remaining_budget <= 0:
                break
            
            # Adding momentum term to the local optimization
            momentum = np.random.uniform(low=-0.1, high=0.1, size=self.dim)
            initial_guess_with_momentum = initial_guess + momentum
            
            result = minimize(func, initial_guess_with_momentum, method='L-BFGS-B', bounds=bounds, options={'maxfun': remaining_budget})
            remaining_budget -= result.nfev
            
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
        
        return best_solution

# Example usage:
# Assuming you have a function `func` with attributes `bounds.lb` and `bounds.ub`
# optimizer = HybridLocalGlobalOptimizer(budget=100, dim=2)
# best_solution = optimizer(func)