import numpy as np
from scipy.optimize import minimize

class AdaptiveLocalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        best_solution = None
        best_value = float('inf')
        evaluations = 0

        while evaluations < self.budget:
            initial_guess = np.random.uniform(bounds[:, 0], bounds[:, 1], size=self.dim)
            result = minimize(func, initial_guess, method='L-BFGS-B', bounds=bounds)

            evaluations += result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            if evaluations >= self.budget:
                break

            shrink_factor = 0.9 * (1 - evaluations / self.budget)  # Adaptive shrink factor
            bounds = np.array([
                np.maximum(func.bounds.lb, best_solution - shrink_factor * (best_solution - func.bounds.lb)),
                np.minimum(func.bounds.ub, best_solution + shrink_factor * (func.bounds.ub - best_solution))
            ]).T

        return best_solution, best_value