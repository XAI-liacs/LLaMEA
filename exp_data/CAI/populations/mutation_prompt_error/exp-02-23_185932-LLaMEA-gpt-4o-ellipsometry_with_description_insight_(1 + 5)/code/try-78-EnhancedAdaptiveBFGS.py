import numpy as np
from scipy.optimize import minimize

class EnhancedAdaptiveBFGS:
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
        initial_guesses = np.random.uniform(bounds[0] + 0.07 * (bounds[1] - bounds[0]),  # Adjusted line
                                            bounds[1] - 0.07 * (bounds[1] - bounds[0]),  # Adjusted line
                                            (num_initial_guesses, self.dim))

        for guess in initial_guesses:
            result = minimize(self.evaluate_func, guess, args=(func,),
                              method='L-BFGS-B', bounds=bounds.T,
                              options={'maxfun': self.budget - self.evaluations})

            if result.success and result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            
            # Break early if budget exceeded
            if self.evaluations >= self.budget:
                break

            # Adaptive boundary adjustment with dynamic constraint relaxation
            bounds_range = 0.1 * (bounds[1] - bounds[0])
            bounds[0] = np.maximum(func.bounds.lb, best_solution - bounds_range)
            bounds[1] = np.minimum(func.bounds.ub, best_solution + bounds_range)

        return best_solution

    def evaluate_func(self, x, func):
        if self.evaluations < self.budget:
            value = func(x)
            self.evaluations += 1
            return value
        else:
            # Avoid further evaluations and terminate early
            return float('inf')