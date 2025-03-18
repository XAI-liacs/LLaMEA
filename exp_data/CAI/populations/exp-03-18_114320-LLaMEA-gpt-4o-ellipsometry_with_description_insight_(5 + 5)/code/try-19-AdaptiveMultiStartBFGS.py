import numpy as np
from scipy.optimize import minimize

class AdaptiveMultiStartBFGS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        best_solution = None
        best_value = np.inf
        
        initial_budget_ratio = 0.35  # Changed line
        initial_budget = int(self.budget * initial_budget_ratio)
        refinement_budget = self.budget - initial_budget
        
        # Enhanced diversity in sampling
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        initial_guesses = np.random.uniform(bounds[:, 0], bounds[:, 1], (initial_budget, self.dim))
        
        for initial_guess in initial_guesses:
            if self.evaluations >= self.budget:
                break
            result = minimize(func, initial_guess, method='L-BFGS-B', bounds=bounds, options={'maxiter': refinement_budget})
            self.evaluations += result.nfev
            
            # Check for better solution
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
        
        return best_solution

# Example usage:
# optimizer = AdaptiveMultiStartBFGS(budget=100, dim=2)
# best_solution = optimizer(some_black_box_function)