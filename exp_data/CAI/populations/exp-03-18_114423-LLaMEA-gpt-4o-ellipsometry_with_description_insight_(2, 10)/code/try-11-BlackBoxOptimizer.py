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

        while remaining_budget > 0:
            # Uniform sampling within bounds for initial guess
            initial_guess = np.array([np.random.uniform(low, high) for low, high in bounds])
            # Adaptive local budget based on remaining evaluations
            local_budget = min(remaining_budget, max(10, remaining_budget // 5))

            # BFGS optimization with bounds and limited evaluations
            result = minimize(func, initial_guess, method='L-BFGS-B', bounds=bounds, options={'maxfun': local_budget})

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
                # Dynamic bounds adjustment around best solution
                bounds = [(max(low, x - 10), min(high, x + 10)) for (low, high), x in zip(bounds, best_solution)]

            remaining_budget -= result.nfev

        return best_solution, best_value