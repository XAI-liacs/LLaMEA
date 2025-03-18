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

        # Local optimization using BFGS
        bounds = [(lb[i], ub[i]) for i in range(self.dim)]
        remaining_evaluations = remaining_budget // self.dim

        def wrapped_func(x):
            nonlocal remaining_evaluations
            if remaining_evaluations <= 0:
                raise StopIteration("Budget exceeded")
            remaining_evaluations -= 1
            return func(x)

        # Use BFGS with modified stopping criterion for local refinement
        best_local_sample, best_local_value = best_sample, best_value
        for _ in range(2):  # Strategic restart mechanism
            result = minimize(
                wrapped_func,
                best_local_sample,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxfun': remaining_evaluations, 'ftol': 1e-12} # Modified tolerance
            )
            if result.fun < best_local_value:
                best_local_value = result.fun
                best_local_sample = result.x

        return best_local_sample, best_local_value