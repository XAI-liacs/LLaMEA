import numpy as np
from scipy.optimize import minimize

class AdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initial uniform sampling across the parameter space
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        num_initial_samples = min(10, self.budget // 3)  # Adjusted initial samples for dynamic sampling
        initial_samples = np.random.uniform(func.bounds.lb, func.bounds.ub, (num_initial_samples, self.dim))
        evals = 0
        best_sample = None
        best_value = float('inf')

        # Evaluate initial samples
        for sample in initial_samples:
            if evals >= self.budget:
                break
            value = func(sample)
            evals += 1
            if value < best_value:
                best_value = value
                best_sample = sample

        # Local optimization using BFGS starting from the best initial sample
        if evals < self.budget:
            def wrapped_func(x):
                nonlocal evals
                if evals >= self.budget:
                    return float('inf')
                value = func(x)
                evals += 1
                return value

            # Adjust starting point slightly for a more robust local search
            perturbed_best_sample = best_sample + np.random.normal(0, 0.01, size=self.dim)
            result = minimize(wrapped_func, perturbed_best_sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - evals})
            if result.fun < best_value:
                best_value = result.fun
                best_sample = result.x

        return best_sample