import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Adaptive initial sampling based on dimension
        initial_samples = int(self.budget * (0.05 + 0.05 * (3 / self.dim)))
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

        # Local optimization using Nelder-Mead for smooth convergence
        bounds = [(lb[i], ub[i]) for i in range(self.dim)]
        remaining_evaluations = remaining_budget // self.dim

        def wrapped_func(x):
            nonlocal remaining_evaluations
            if remaining_evaluations <= 0:
                raise StopIteration("Budget exceeded")
            remaining_evaluations -= 1
            return func(x)

        # Choose optimization method based on remaining budget
        if remaining_budget > 50:
            result = minimize(
                wrapped_func,
                best_sample,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxfun': remaining_budget, 'ftol': 1e-12}
            )
        else:
            result = minimize(
                wrapped_func,
                best_sample,
                method='Nelder-Mead',
                options={'maxfev': remaining_budget, 'xatol': 1e-8}
            )

        return result.x, result.fun