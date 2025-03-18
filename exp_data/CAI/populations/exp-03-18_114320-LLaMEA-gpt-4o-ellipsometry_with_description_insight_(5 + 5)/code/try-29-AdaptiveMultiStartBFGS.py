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
        
        initial_budget_ratio = 0.35
        initial_budget = int(self.budget * initial_budget_ratio)
        refinement_budget = self.budget - initial_budget
        
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        initial_guesses = np.random.uniform(bounds[:, 0], bounds[:, 1], (initial_budget, self.dim))
        no_improvement_count = 0  # Added line
        dynamic_budget_factor = 0.1  # New variable to dynamically adjust budget
        
        for initial_guess in initial_guesses:
            if self.evaluations >= self.budget:
                break
            current_refinement_budget = int(refinement_budget * (1 + dynamic_budget_factor))  # Adjusted line
            result = minimize(func, initial_guess, method='L-BFGS-B', bounds=bounds, options={'maxiter': current_refinement_budget})
            self.evaluations += result.nfev
            
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
                no_improvement_count = 0  # Reset if improvement is found
                dynamic_budget_factor = max(0, dynamic_budget_factor - 0.05)  # Reduce factor on improvement
            else:
                no_improvement_count += 1  # Added line
                dynamic_budget_factor = min(0.2, dynamic_budget_factor + 0.05)  # Increase factor on no improvement
            
            if no_improvement_count > 5:  # Added line
                initial_guesses = np.random.uniform(bounds[:, 0], bounds[:, 1], (2, self.dim))  # Restart with fewer guesses
                no_improvement_count = 0  # Reset counter after restart

        return best_solution