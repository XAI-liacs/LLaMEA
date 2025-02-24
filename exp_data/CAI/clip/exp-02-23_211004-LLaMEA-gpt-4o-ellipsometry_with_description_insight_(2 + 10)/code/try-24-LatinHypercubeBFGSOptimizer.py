import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class LatinHypercubeBFGSOptimizer:
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

        # Use Latin Hypercube Sampling for initial exploration
        num_initial_samples = min(15, self.budget // 10)  # Use a portion of the budget
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(num_initial_samples)
        initial_guesses = qmc.scale(sample, lb, ub)

        best_result = None

        # Evaluate initial points and keep the best using BFGS
        for initial_guess in initial_guesses:
            try:
                result = minimize(budgeted_func, initial_guess, method='BFGS', options={'gtol': 1e-8})  # Precision-targeted restart
                if best_result is None or result.fun < best_result.fun:
                    best_result = result
            except RuntimeError:
                break

        return best_result.x if best_result else None