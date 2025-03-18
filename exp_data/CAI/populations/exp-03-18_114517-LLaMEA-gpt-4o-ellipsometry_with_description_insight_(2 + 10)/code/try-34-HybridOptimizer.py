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

        # Uniform sampling for initial guesses
        lb, ub = func.bounds.lb, func.bounds.ub
        samples = np.random.uniform(lb, ub, (initial_samples, self.dim))
        best_sample, best_value = None, float('inf')

        for sample in samples:
            value = func(sample)
            if value < best_value:
                best_value = value
                best_sample = sample

        # Local optimization using BFGS and Nelder-Mead
        bounds = [(lb[i], ub[i]) for i in range(self.dim)]
        remaining_evaluations = remaining_budget // (self.dim * 2)  # Adjusted for dual local optimizers

        def wrapped_func(x):
            nonlocal remaining_evaluations
            if remaining_evaluations <= 0:
                raise StopIteration("Budget exceeded")
            remaining_evaluations -= 1
            return func(x)

        # Use BFGS with modified stopping criterion for local refinement
        result_bfgs = minimize(
            wrapped_func,
            best_sample,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxfun': remaining_budget//2, 'ftol': 1e-8} # Modified tolerance for precision
        )

        # Additional Nelder-Mead optimization for robustness
        result_nm = minimize(
            wrapped_func,
            best_sample,
            method='Nelder-Mead',
            options={'maxfev': remaining_budget//2}  # Adjusted max function evaluations
        )

        # Return the best result from both local optimizations
        if result_bfgs.fun < result_nm.fun:
            return result_bfgs.x, result_bfgs.fun
        else:
            return result_nm.x, result_nm.fun