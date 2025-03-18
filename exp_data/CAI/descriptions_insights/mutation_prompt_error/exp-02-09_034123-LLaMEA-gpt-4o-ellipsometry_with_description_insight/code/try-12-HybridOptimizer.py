import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        num_samples = max(1, int(5 * self.dim * (self.budget / (10 * self.dim))))  # Dynamic adjustment
        initial_samples = self._uniform_sampling(bounds, num_samples)
        best_sample = None
        best_value = float('inf')
        
        for sample in initial_samples:
            if self.budget <= 0:
                break
            res = minimize(func, sample, method='L-BFGS-B', bounds=bounds, options={'gtol': 1e-6}) # Changed line
            self.budget -= res.nfev  # Ensure budget is decremented by the number of function evaluations
            if res.fun < best_value:
                best_value = res.fun
                best_sample = res.x
            bounds = np.clip(bounds * 0.95 + best_sample * 0.05, func.bounds.lb, func.bounds.ub)  # Adaptive narrowing
            
        return best_sample

    def _uniform_sampling(self, bounds, num_samples):
        samples = []
        for _ in range(num_samples):
            sample = np.array([np.random.uniform(low, high) for low, high in bounds])
            samples.append(sample)
        return samples