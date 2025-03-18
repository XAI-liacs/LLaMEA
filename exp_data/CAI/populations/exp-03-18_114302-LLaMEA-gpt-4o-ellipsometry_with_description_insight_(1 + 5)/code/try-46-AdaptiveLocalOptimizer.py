import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class AdaptiveLocalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine the bounds from the function
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        best_solution = None
        best_value = float('inf')
        evaluations = 0

        # Main loop of the algorithm
        while evaluations < self.budget:
            # Uniform sampling for initial guess using Latin Hypercube Sampling
            sampler = qmc.LatinHypercube(d=self.dim)  # New line
            initial_guess = sampler.random(n=1).flatten() * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]  # New line

            # Local optimization using BFGS
            result = minimize(func, initial_guess, method='L-BFGS-B', bounds=bounds)

            evaluations += result.nfev

            # Recording the best solution found
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            if evaluations >= self.budget:
                break

            # Adjust the search space bounds based on current best solution
            shrink_factor = 0.9
            bounds = np.array([
                np.maximum(func.bounds.lb, best_solution - shrink_factor * (best_solution - func.bounds.lb)),
                np.minimum(func.bounds.ub, best_solution + shrink_factor * (func.bounds.ub - best_solution))
            ]).T

        return best_solution, best_value