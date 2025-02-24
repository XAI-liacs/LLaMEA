import numpy as np
from scipy.optimize import minimize

class AdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initial uniform sampling across the parameter space
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        num_initial_samples = min(10, self.budget // 2)  # Limit initial samples to a reasonable number

        # Change made here: Use centroid of bounds as a seed for better initial guess
        centroid_seed = (func.bounds.lb + func.bounds.ub) / 2
        initial_samples = np.vstack((np.random.uniform(func.bounds.lb, func.bounds.ub, (num_initial_samples - 1, self.dim)), centroid_seed))
        
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

        # Augment samples dynamically if budget allows
        if evals < self.budget:
            extra_samples = num_initial_samples // 2
            dynamic_samples = np.random.uniform(func.bounds.lb, func.bounds.ub, (extra_samples, self.dim))
            for sample in dynamic_samples:
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

            result = minimize(wrapped_func, best_sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - evals})
            if result.fun < best_value:
                best_value = result.fun
                best_sample = result.x

        return best_sample