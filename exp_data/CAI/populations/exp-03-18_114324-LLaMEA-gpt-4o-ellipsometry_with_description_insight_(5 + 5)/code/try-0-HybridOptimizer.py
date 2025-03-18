import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initial uniform sampling
        lb, ub = func.bounds.lb, func.bounds.ub
        samples = lb + np.random.rand(self.dim, self.budget // 5) * (ub - lb)
        best_sample = None
        best_value = float('inf')

        # Evaluate initial samples
        for i in range(samples.shape[1]):
            if self.budget <= 0:
                break
            value = func(samples[:, i])
            self.budget -= 1
            if value < best_value:
                best_value = value
                best_sample = samples[:, i]

        # Use BFGS for local optimization from the best initial sample
        if best_sample is not None:
            result = minimize(
                func, 
                best_sample, 
                method='L-BFGS-B', 
                bounds=[(lb[i], ub[i]) for i in range(self.dim)],
                options={'maxfun': self.budget}
            )
            if result.success and result.fun < best_value:
                best_value = result.fun
                best_sample = result.x

        return best_sample