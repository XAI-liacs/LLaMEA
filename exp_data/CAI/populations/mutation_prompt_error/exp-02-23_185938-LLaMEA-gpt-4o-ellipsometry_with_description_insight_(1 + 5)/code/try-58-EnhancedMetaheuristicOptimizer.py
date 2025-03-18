import numpy as np
from scipy.optimize import minimize

class EnhancedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        initial_sample_count = max(12, self.budget // 10)  # Adjust initial samples to balance exploration
        initial_samples = []
        for _ in range(initial_sample_count):
            sample = np.array([
                np.random.uniform(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)
            ])
            initial_samples.append(sample)

        best_sample = None
        best_value = float('inf')
        for sample in initial_samples:
            value = func(sample)
            self.budget -= 1
            if value < best_value:
                best_value = value
                best_sample = sample
            if self.budget <= 0:
                return best_sample

        bounds = [(max(lb, x - 0.2 * (ub - lb)), min(ub, x + 0.2 * (ub - lb)))  # Adaptive expansion of bounds
                  for x, lb, ub in zip(best_sample, func.bounds.lb, func.bounds.ub)]

        def objective(x):
            return func(x)

        res = minimize(objective, x0=best_sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': int(self.budget * 0.8), 'ftol': 1e-9})  # Adjust ftol for precision

        if res.success:
            return res.x
        else:
            return best_sample