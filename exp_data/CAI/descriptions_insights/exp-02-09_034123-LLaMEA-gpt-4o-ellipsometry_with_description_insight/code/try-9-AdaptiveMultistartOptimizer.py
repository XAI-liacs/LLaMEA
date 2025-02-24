import numpy as np
from scipy.optimize import minimize

class AdaptiveMultistartOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        num_starts = max(1, int(3 * self.dim * (self.budget / (15 * self.dim))))  # Adjust number of restarts
        initial_samples = self._uniform_sampling(bounds, num_starts)
        best_sample = None
        best_value = float('inf')
        
        for sample in initial_samples:
            if self.budget <= 0:
                break
            # Optimize with L-BFGS-B method
            res = minimize(func, sample, method='L-BFGS-B', bounds=bounds)
            self.budget -= res.nfev  # Deduct budget by number of function evaluations
            if res.fun < best_value:
                best_value = res.fun
                best_sample = res.x
                # Iteratively refine the region around the best sample
                refined_bounds = self._refine_bounds(best_sample, bounds, factor=0.85)
                res_refined = minimize(func, best_sample, method='L-BFGS-B', bounds=refined_bounds)
                self.budget -= res_refined.nfev
                if res_refined.fun < best_value:
                    best_value = res_refined.fun
                    best_sample = res_refined.x

        return best_sample

    def _uniform_sampling(self, bounds, num_samples):
        return [np.random.uniform(low, high, size=self.dim) for low, high in bounds]

    def _refine_bounds(self, best_sample, bounds, factor):
        return np.clip(bounds * factor + best_sample * (1 - factor), func.bounds.lb, func.bounds.ub)