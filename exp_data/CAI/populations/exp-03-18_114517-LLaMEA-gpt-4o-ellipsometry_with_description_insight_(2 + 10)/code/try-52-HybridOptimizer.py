import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Ensure initial budget usage for a global search
        initial_samples = int(self.budget * 0.1)
        remaining_budget = self.budget - initial_samples

        # Adaptive mutation for initial guesses
        lb, ub = func.bounds.lb, func.bounds.ub
        mutation_rate = (ub - lb) * 0.1
        samples = np.random.uniform(lb, ub, (initial_samples, self.dim)) + np.random.uniform(-mutation_rate, mutation_rate, (initial_samples, self.dim))
        best_sample, best_value = None, float('inf')

        for sample in samples:
            value = func(sample)
            if value < best_value:
                best_value = value
                best_sample = sample

        # Local optimization using dual methods
        bounds = [(lb[i], ub[i]) for i in range(self.dim)]
        remaining_evaluations = remaining_budget // self.dim

        def wrapped_func(x):
            nonlocal remaining_evaluations
            if remaining_evaluations <= 0:
                raise StopIteration("Budget exceeded")
            remaining_evaluations -= 1
            return func(x)

        # Use BFGS and Nelder-Mead for local refinement
        result_bfgs = minimize(
            wrapped_func,
            best_sample,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxfun': remaining_budget // 2, 'ftol': 1e-12}
        )
        result_nelder = minimize(
            wrapped_func,
            best_sample,
            method='Nelder-Mead',
            bounds=bounds,
            options={'maxfev': remaining_budget // 2, 'fatol': 1e-12}
        )

        # Choose the best result from both methods
        if result_bfgs.fun < result_nelder.fun:
            return result_bfgs.x, result_bfgs.fun
        else:
            return result_nelder.x, result_nelder.fun