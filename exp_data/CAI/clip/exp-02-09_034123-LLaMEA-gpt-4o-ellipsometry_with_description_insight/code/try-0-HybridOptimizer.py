import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        initial_samples = self._uniform_sampling(bounds, 5 * self.dim)
        best_sample = None
        best_value = float('inf')
        
        for sample in initial_samples:
            res = minimize(func, sample, method='L-BFGS-B', bounds=bounds)
            if res.fun < best_value:
                best_value = res.fun
                best_sample = res.x
            
            if self.budget <= 0:
                break
        
        return best_sample

    def _uniform_sampling(self, bounds, num_samples):
        samples = []
        for _ in range(num_samples):
            sample = np.array([np.random.uniform(low, high) for low, high in bounds])
            samples.append(sample)
        return samples