import numpy as np
from scipy.optimize import minimize

class AdaptiveBFGS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        best_solution = None
        best_value = float('inf')

        # Uniform sampling for initial guesses
        num_initial_guesses = min(5, self.budget // self.dim)
        initial_guesses = np.random.uniform(bounds[0], bounds[1], (num_initial_guesses, self.dim))

        for guess in initial_guesses:
            remaining_budget = self.budget - self.evaluations
            if remaining_budget <= 0:
                break

            result = minimize(self.evaluate_func, guess, args=(func,),
                              method='L-BFGS-B', bounds=bounds.T,
                              options={'maxfun': remaining_budget})

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            # Adaptive boundary adjustment
            bounds[0] = np.maximum(bounds[0], best_solution - 0.1 * (bounds[1] - bounds[0]))
            bounds[1] = np.minimum(bounds[1], best_solution + 0.1 * (bounds[1] - bounds[0]))

        return best_solution

    def evaluate_func(self, x, func):
        if self.evaluations < self.budget:
            value = func(x)
            self.evaluations += 1
            return value
        else:
            raise RuntimeError("Budget exceeded")