import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        best_solution = None
        best_value = np.inf
        remaining_budget = self.budget

        adaptive_factor = 0.8  # New adaptive factor for dynamic adjustment

        while remaining_budget > 0:
            # Uniform sampling with an adaptive step size for initial guess
            initial_guess = np.array([low + (high - low) * (np.random.rand() ** adaptive_factor) for low, high in bounds])
            
            # Adjust local budget based on adaptive factor
            local_budget = int(max(10, remaining_budget * adaptive_factor))
            
            # BFGS optimization with bounds and limited evaluations
            result = minimize(func, initial_guess, method='L-BFGS-B', bounds=bounds, options={'maxfun': local_budget})

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            remaining_budget -= min(local_budget, result.nfev)  # Adjusting budget decrement

        return best_solution, best_value