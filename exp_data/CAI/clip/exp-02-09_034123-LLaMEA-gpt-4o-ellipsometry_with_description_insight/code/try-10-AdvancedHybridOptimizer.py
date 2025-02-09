import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

class AdvancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        num_samples = max(1, int(5 * self.dim * (self.budget / (10 * self.dim))))
        initial_samples = self._bayesian_sampling(bounds, num_samples)
        best_sample = None
        best_value = float('inf')
        
        for sample in initial_samples:
            if self.budget <= 0:
                break
            res = minimize(func, sample, method='L-BFGS-B', bounds=bounds)
            self.budget -= res.nfev
            if res.fun < best_value:
                best_value = res.fun
                best_sample = res.x
            bounds = self._adaptive_narrowing(bounds, best_sample, func.bounds)
            
        return best_sample

    def _bayesian_sampling(self, bounds, num_samples):
        samples = []
        for _ in range(num_samples):
            sample = np.array([norm.rvs(loc=(low + high) / 2, scale=(high - low) / 4) for low, high in bounds])
            sample = np.clip(sample, bounds[:,0], bounds[:,1])
            samples.append(sample)
        return samples

    def _adaptive_narrowing(self, bounds, best_sample, original_bounds):
        new_bounds = bounds * 0.95 + best_sample * 0.05
        return np.clip(new_bounds, original_bounds.lb, original_bounds.ub)