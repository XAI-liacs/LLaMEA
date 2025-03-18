import numpy as np
from scipy.optimize import minimize

class AdaptiveLocalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory = []  # Memory to keep track of best solutions found

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        best_solution = None
        best_value = float('inf')
        evaluations = 0

        while evaluations < self.budget:
            if self.memory:
                initial_guess = np.mean(self.memory, axis=0)  # Use average of memory
                initial_guess += np.random.normal(0, 0.1, self.dim)  # Add perturbation
            else:
                initial_guess = np.random.uniform(bounds[:, 0], bounds[:, 1], size=self.dim)

            result = minimize(func, initial_guess, method='L-BFGS-B', bounds=bounds)
            evaluations += result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
                self.memory.append(best_solution)
                self.memory = self.memory[-5:]  # Keep only recent bests

            if evaluations >= self.budget:
                break

            shrink_factor = 0.75
            bounds = np.array([
                np.maximum(func.bounds.lb, best_solution - shrink_factor * (best_solution - func.bounds.lb)),
                np.minimum(func.bounds.ub, best_solution + shrink_factor * (func.bounds.ub - best_solution))
            ]).T

            if evaluations < self.budget / 2:  
                grid_points = np.linspace(bounds[:, 0], bounds[:, 1], num=3).T
                additional_guesses = [np.random.choice(gp, size=self.dim) for gp in grid_points]
                for guess in additional_guesses:
                    if evaluations >= self.budget:
                        break
                    result = minimize(func, guess, method='L-BFGS-B', bounds=bounds)
                    evaluations += result.nfev
                    if result.fun < best_value:
                        best_value = result.fun
                        best_solution = result.x
                        self.memory.append(best_solution)

        return best_solution, best_value