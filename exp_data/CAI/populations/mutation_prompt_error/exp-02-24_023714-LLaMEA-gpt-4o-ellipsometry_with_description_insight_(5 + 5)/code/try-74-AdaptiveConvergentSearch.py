import numpy as np
from scipy.optimize import minimize

class AdaptiveConvergentSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluation_count = 0

    def __call__(self, func):
        # Initial adaptive sampling with higher weight to mid-range values
        initial_guesses = [np.random.uniform(func.bounds.lb, func.bounds.ub) * 0.5 + 0.5 * (func.bounds.lb + func.bounds.ub) / 2 for _ in range(10)]
        best_solution = None
        best_value = float('inf')

        # Define a wrapping function to count evaluations
        def wrapped_func(x):
            if self.evaluation_count >= self.budget:
                return float('inf')
            self.evaluation_count += 1
            return func(x)

        # Begin with a local optimizer
        for guess in initial_guesses:
            # Dynamically select the optimization method
            method = 'L-BFGS-B' if self.evaluation_count < self.budget / 2 else 'Nelder-Mead'
            result = minimize(wrapped_func, guess, method=method, bounds=list(zip(func.bounds.lb, func.bounds.ub)))

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            # Update bounds based on the current best solution
            func.bounds.lb = np.maximum(func.bounds.lb, best_solution - 0.1 * (np.array(func.bounds.ub) - np.array(func.bounds.lb)))
            func.bounds.ub = np.minimum(func.bounds.ub, best_solution + 0.1 * (np.array(func.bounds.ub) - np.array(func.bounds.lb)))

            # If budget is exhausted, terminate
            if self.evaluation_count >= self.budget:
                break

        return best_solution