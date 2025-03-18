import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        initial_samples = int(self.budget * 0.15)  # Adjusted initial sample percentage
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
        remaining_evaluations = remaining_budget // self.dim

        def wrapped_func(x):
            nonlocal remaining_evaluations
            if remaining_evaluations <= 0:
                raise StopIteration("Budget exceeded")
            remaining_evaluations -= 1
            return func(x)

        def dynamic_bounds_adjustment(bounds, best_sample, shrink_factor=0.9):
            return [(max(b[0], best_sample[i] - (b[1] - b[0]) * shrink_factor), 
                     min(b[1], best_sample[i] + (b[1] - b[0]) * shrink_factor)) for i, b in enumerate(bounds)]

        bounds = dynamic_bounds_adjustment(bounds, best_sample)

        result = minimize(
            wrapped_func,
            best_sample,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxfun': remaining_budget, 'ftol': 1e-9}
        )

        return result.x, result.fun