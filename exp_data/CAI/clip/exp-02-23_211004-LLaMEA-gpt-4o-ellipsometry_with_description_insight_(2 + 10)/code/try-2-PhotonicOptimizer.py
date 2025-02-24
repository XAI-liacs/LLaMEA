import numpy as np
from scipy.optimize import minimize

class PhotonicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Keep track of the number of function evaluations
        evaluations = 0

        # Define the bounds for the search space
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)

        # Function to count function evaluations and check budget
        def budgeted_func(x):
            nonlocal evaluations
            if evaluations >= self.budget:
                raise RuntimeError("Exceeded budget of function evaluations.")
            evaluations += 1
            return func(x)

        # Initial weighted random sampling to generate initial guesses
        num_initial_samples = min(10, self.budget // 10)  # Use a portion of the budget
        weights = np.linspace(1, 0, num_initial_samples)  # Decreasing weights
        initial_guesses = lb + (ub - lb) * np.random.choice(np.random.rand(num_initial_samples, self.dim), p=weights/weights.sum())

        best_result = None

        # Evaluate initial points and keep the best
        for initial_guess in initial_guesses:
            try:
                result = minimize(budgeted_func, initial_guess, method='Nelder-Mead')
                if best_result is None or result.fun < best_result.fun:
                    best_result = result
            except RuntimeError:
                break

        return best_result.x if best_result else None