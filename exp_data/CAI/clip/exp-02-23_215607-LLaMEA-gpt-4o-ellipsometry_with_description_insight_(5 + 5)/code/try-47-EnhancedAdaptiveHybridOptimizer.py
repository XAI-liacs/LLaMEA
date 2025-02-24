import numpy as np
from scipy.optimize import minimize

class EnhancedAdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initial uniform sampling across the parameter space
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        num_initial_samples = min(15, self.budget // 3)  # Adjusted initial sampling size
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
        
        # Modified to prioritize regions with higher sample variance
        variances = [np.var(initial_samples[:, i]) for i in range(self.dim)]
        prioritized_sample = initial_samples[np.argmax(variances)]
        best_sample = prioritized_sample if best_value == float('inf') else best_sample

        # Local optimization using BFGS with adaptive learning rate
        if evals < self.budget:
            def wrapped_func(x):
                nonlocal evals
                if evals >= self.budget or best_value < 1e-6:  # Adjusted convergence threshold
                    return float('inf')
                value = func(x)
                evals += 1
                return value

            result = minimize(
                wrapped_func, 
                best_sample, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxfun': self.budget - evals, 'adaptive': True}
            )
            
            if result.success and result.fun < best_value:
                best_value = result.fun
                best_sample = result.x

        return best_sample