import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class LatinHypercubeBFGSRefinedOptimizer:
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

        # Dynamic budget allocation
        initial_sample_budget = max(5, self.budget // 20)
        refine_budget = self.budget - initial_sample_budget

        # Use Latin Hypercube Sampling for initial exploration
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(initial_sample_budget)
        initial_guesses = qmc.scale(sample, lb, ub)

        best_result = None

        # Evaluate initial points to find the best initial guess
        for initial_guess in initial_guesses:
            try:
                current_value = budgeted_func(initial_guess)
                if best_result is None or current_value < best_result.fun:
                    best_result = type('Result', (), {'x': initial_guess, 'fun': current_value})
            except RuntimeError:
                break

        # Further refine the best initial guess using BFGS
        if best_result:
            try:
                result = minimize(budgeted_func, best_result.x, method='BFGS', options={'maxiter': refine_budget})
                if result.fun < best_result.fun:
                    best_result = result
            except RuntimeError:
                pass

        return best_result.x if best_result else None