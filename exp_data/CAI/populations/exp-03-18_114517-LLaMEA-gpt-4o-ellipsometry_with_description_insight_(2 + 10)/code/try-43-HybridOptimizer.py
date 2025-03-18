import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        initial_samples = int(self.budget * 0.15) 
        remaining_budget = self.budget - initial_samples

        lb, ub = func.bounds.lb, func.bounds.ub
        samples = np.random.uniform(lb, ub, (initial_samples, self.dim))
        best_sample, best_value = None, float('inf')

        for sample in samples:
            value = func(sample)
            if value < best_value:
                best_value = value
                best_sample = sample

        bounds = [(lb[i], ub[i]) for i in range(self.dim)]
        
        def wrapped_func(x):
            nonlocal remaining_budget
            if remaining_budget <= 0:
                raise StopIteration("Budget exceeded")
            remaining_budget -= 1
            return func(x)

        # Adaptive local optimization using trust-region method
        trust_region_options = {'initial_trust_radius': 0.1, 'max_trust_radius': 1.0}
        result = minimize(
            wrapped_func,
            best_sample,
            method='trust-constr',
            bounds=bounds,
            options={'maxiter': remaining_budget, 'gtol': 1e-6, **trust_region_options}
        )

        return result.x, result.fun