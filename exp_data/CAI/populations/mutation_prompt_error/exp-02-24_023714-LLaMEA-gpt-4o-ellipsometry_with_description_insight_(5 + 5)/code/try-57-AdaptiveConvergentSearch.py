import numpy as np
from scipy.optimize import minimize

class AdaptiveConvergentSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluation_count = 0

    def __call__(self, func):
        # Initial uniform sampling for good coverage
        initial_guesses = [np.random.uniform(func.bounds.lb, func.bounds.ub) for _ in range(10)]
        best_solution = None
        best_value = float('inf')
        temperature = 1.0  # Initial temperature for acceptance criterion

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

            # Acceptance with temperature-based criterion
            if result.fun < best_value or np.exp((best_value - result.fun) / temperature) > np.random.rand():
                best_value = result.fun
                best_solution = result.x

            # Update bounds based on the current best solution
            func.bounds.lb = np.maximum(func.bounds.lb, best_solution - 0.1 * (np.array(func.bounds.ub) - np.array(func.bounds.lb)))
            func.bounds.ub = np.minimum(func.bounds.ub, best_solution + 0.1 * (np.array(func.bounds.ub) - np.array(func.bounds.lb)))

            # If budget is exhausted, terminate
            if self.evaluation_count >= self.budget:
                break

            # Cool down the temperature
            temperature *= 0.95

        return best_solution