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

        # Gaussian sampling for initial guesses
        num_initial_guesses = min(5, self.budget // self.dim)
        mean = (bounds[0] + bounds[1]) / 2
        std_dev = (bounds[1] - bounds[0]) / 6  # Assuming 3 std devs cover the range
        initial_guesses = np.clip(np.random.normal(mean, std_dev, (num_initial_guesses, self.dim)),
                                  bounds[0], bounds[1])

        for guess in initial_guesses:
            adaptive_budget = self.budget - self.evaluations - num_initial_guesses  # Adjusted line
            result = minimize(self.evaluate_func, guess, args=(func,),
                              method='L-BFGS-B', bounds=bounds.T,
                              options={'maxfun': adaptive_budget})  # Adjusted line

            if result.success and result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            
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
            return float('inf')