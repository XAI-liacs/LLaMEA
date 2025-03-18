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
        remaining_evaluations = remaining_budget // (self.dim * 2)  # Changed multiplier for local refinement

        def wrapped_func(x):
            nonlocal remaining_evaluations
            if remaining_evaluations <= 0:
                raise StopIteration("Budget exceeded")
            remaining_evaluations -= 1
            return func(x)

        # Multi-start local refinement
        result = None
        for _ in range(2):  # Added multi-start loop
            temp_result = minimize(
                wrapped_func,
                best_sample,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxfun': remaining_budget, 'ftol': 1e-9}
            )
            if result is None or temp_result.fun < result.fun:
                result = temp_result

        return result.x, result.fun