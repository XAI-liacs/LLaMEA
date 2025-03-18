import numpy as np
from scipy.optimize import minimize

class AdaptiveHybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        bounds = func.bounds
        lower_bound = bounds.lb
        upper_bound = bounds.ub

        # Adaptive bounds adjustment parameters
        adjustment_factor = 0.15  # Changed from 0.1 to 0.15 for improved convergence
        convergence_threshold = 1e-5

        # Uniformly sample initial points
        initial_guess = lower_bound + np.random.rand(self.dim) * (upper_bound - lower_bound)
        best_solution = initial_guess
        best_value = func(initial_guess)
        self.evaluations += 1

        # Local optimization using BFGS
        def local_optimize(x0):
            nonlocal best_solution, best_value
            result = minimize(func, x0, method='L-BFGS-B', bounds=list(zip(lower_bound, upper_bound)))
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        while self.evaluations < self.budget:
            # Perform local search
            local_optimize(best_solution)

            # Dynamically adjust bounds
            for i in range(self.dim):
                lower_bound[i] = max(bounds.lb[i], best_solution[i] - adjustment_factor * (upper_bound[i] - lower_bound[i]))
                upper_bound[i] = min(bounds.ub[i], best_solution[i] + adjustment_factor * (upper_bound[i] - lower_bound[i]))

            # Convergence check
            if np.linalg.norm(best_solution - initial_guess) < convergence_threshold:
                break

            initial_guess = best_solution
            self.evaluations += 1

        return best_solution