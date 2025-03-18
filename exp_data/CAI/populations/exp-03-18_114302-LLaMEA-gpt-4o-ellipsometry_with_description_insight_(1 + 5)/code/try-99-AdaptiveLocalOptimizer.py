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
            if best_solution is None or np.random.rand() < 0.1:  # Change 1
                initial_guess = np.random.uniform(bounds[:, 0], bounds[:, 1], size=self.dim)
            else:
                initial_guess = best_solution + np.random.normal(0, 0.1, size=self.dim)  # Change 2

            result = minimize(func, initial_guess, method='L-BFGS-B', bounds=bounds)
            evaluations += result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            if evaluations >= self.budget:
                break

            shrink_factor = 0.85  # Change 3: Adjusted shrink factor for bounds adaptation
            bounds = np.array([
                np.maximum(func.bounds.lb, best_solution - shrink_factor * (best_solution - func.bounds.lb)),
                np.minimum(func.bounds.ub, best_solution + shrink_factor * (func.bounds.ub - best_solution))
            ]).T

            if evaluations < self.budget / 2:  
                grid_points = np.linspace(bounds[:, 0], bounds[:, 1], num=5).T  # Change 4
                additional_guesses = [np.random.choice(gp, size=self.dim) for gp in grid_points]  # Change 5
                for guess in additional_guesses:
                    if evaluations >= self.budget:
                        break
                    result = minimize(func, guess, method='L-BFGS-B', bounds=bounds)
                    evaluations += result.nfev
                    if result.fun < best_value:
                        best_value = result.fun
                        best_solution = result.x

        return best_solution, best_value