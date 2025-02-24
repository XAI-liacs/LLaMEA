import numpy as np
from scipy.optimize import minimize

class EnhancedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        initial_sample_count = max(10, self.budget // 6)  # Slightly increase the initial sample count

        # Randomly sample initial points within bounds with a refined strategy
        initial_samples = []
        for _ in range(initial_sample_count):
            sample = np.array([
                np.random.uniform(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)
            ])
            initial_samples.append(sample)

        # Evaluate initial samples and find the best one
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

        # Narrow bounds and refine more aggressively with a dynamic shrinking ratio
        shrink_ratio = 0.1 + 0.05 * np.random.rand()  # Introduce randomness to avoid local minima
        bounds = [(max(lb, x - shrink_ratio * (ub - lb)), min(ub, x + shrink_ratio * (ub - lb)))
                  for x, lb, ub in zip(best_sample, func.bounds.lb, func.bounds.ub)]

        # Define the objective function for the local optimizer
        def objective(x):
            return func(x)

        # Use L-BFGS-B for local optimization with adaptive options
        res = minimize(objective, x0=best_sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': int(self.budget * 0.75), 'ftol': 1e-8})  # Adjust budget allocation

        if res.success:
            return res.x
        else:
            return best_sample  # Fallback if optimization fails