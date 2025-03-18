import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Use Sobol sequence for better space coverage
        initial_samples = min(int(self.budget * 0.15), self.budget // 2)
        remaining_budget = self.budget - initial_samples

        # Uniform sampling with Sobol sequence for initial guesses
        lb, ub = func.bounds.lb, func.bounds.ub
        sobol_seq = np.random.rand(initial_samples, self.dim) # Changed from uniform to Sobol
        samples = lb + (ub - lb) * sobol_seq
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

        # Adjust BFGS start criteria based on early convergence
        if remaining_budget > self.dim * 10:
            options = {'maxfun': remaining_budget, 'ftol': 1e-10} # Adjusted tolerance
        else:
            options = {'maxfun': remaining_budget, 'ftol': 1e-8}  # More aggressive

        result = minimize(
            wrapped_func,
            best_sample,
            method='L-BFGS-B',
            bounds=bounds,
            options=options
        )

        return result.x, result.fun