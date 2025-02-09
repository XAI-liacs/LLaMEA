import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        num_samples = max(1, int(8 * self.dim * (self.budget / (10 * self.dim))))  # Adjusted sampling rate
        initial_samples = self._latin_hypercube_sampling(bounds, num_samples)  # Changed sampling method
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
            bounds = np.clip(bounds * 0.90 + best_sample * 0.10, func.bounds.lb, func.bounds.ub)  # Enhanced narrowing
            
        return best_sample

    def _latin_hypercube_sampling(self, bounds, num_samples):  # Implemented Latin Hypercube Sampling
        samples = []
        for dim in range(bounds.shape[0]):
            sample = np.random.permutation(num_samples) / num_samples
            sample = bounds[dim, 0] + sample * (bounds[dim, 1] - bounds[dim, 0])
            samples.append(sample)
        return np.array(samples).T