import numpy as np
from scipy.optimize import minimize

class AdaptiveRestartOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds from the function
        lb, ub = func.bounds.lb, func.bounds.ub
        remaining_budget = self.budget

        best_solution = None
        best_value = float('inf')

        # Adaptive random restart strategy
        while remaining_budget > 0:
            # Randomly sample an initial guess within the current bounds
            initial_guess = np.random.uniform(lb, ub, size=self.dim)

            # Use local optimization (BFGS) from the current initial guess
            def wrapped_func(x):
                nonlocal remaining_budget
                if remaining_budget <= 0:
                    return float('inf')
                value = func(x)
                remaining_budget -= 1
                return value

            result = minimize(wrapped_func, initial_guess, method='L-BFGS-B', bounds=list(zip(lb, ub)))

            # Update the best solution found
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            # Dynamically adjust bounds based on the best solution found
            lb = np.maximum(lb, best_solution - (ub - lb) * 0.1)
            ub = np.minimum(ub, best_solution + (ub - lb) * 0.1)

        return best_solution