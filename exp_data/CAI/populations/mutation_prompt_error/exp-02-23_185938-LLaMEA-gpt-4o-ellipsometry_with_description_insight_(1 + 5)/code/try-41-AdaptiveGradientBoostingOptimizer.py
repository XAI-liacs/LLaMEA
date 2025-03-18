import numpy as np
from scipy.optimize import minimize

class AdaptiveGradientBoostingOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Calculate the number of initial samples, ensuring a portion of the budget is reserved
        initial_sample_count = max(10, self.budget // 10)
        
        # Randomly sample initial points within bounds with a diversification factor
        diversification_factor = 0.25
        initial_samples = []
        for _ in range(initial_sample_count):
            sample = np.array([
                np.random.uniform(lb + diversification_factor * (ub - lb), ub - diversification_factor * (ub - lb))
                for lb, ub in zip(func.bounds.lb, func.bounds.ub)
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

        # Perform a secondary sampling around the best initial sample
        secondary_sample_count = initial_sample_count // 2
        secondary_samples = []
        for _ in range(secondary_sample_count):
            sample = np.array([
                np.random.uniform(max(lb, x - 0.1 * (ub - lb)), min(ub, x + 0.1 * (ub - lb)))
                for x, lb, ub in zip(best_sample, func.bounds.lb, func.bounds.ub)
            ])
            secondary_samples.append(sample)

        for sample in secondary_samples:
            value = func(sample)
            self.budget -= 1
            if value < best_value:
                best_value = value
                best_sample = sample
            if self.budget <= 0:
                return best_sample

        # Define the objective function for the local optimizer
        def objective(x):
            return func(x)

        # Adaptive L-BFGS-B optimization with remaining budget
        bounds = [(max(lb, x - 0.05 * (ub - lb)), min(ub, x + 0.05 * (ub - lb)))
                  for x, lb, ub in zip(best_sample, func.bounds.lb, func.bounds.ub)]
        
        res = minimize(objective, x0=best_sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget, 'ftol': 1e-8})

        if res.success:
            return res.x
        else:
            return best_sample  # Fallback if optimization fails