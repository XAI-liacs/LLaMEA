import numpy as np
from scipy.optimize import minimize

class AdaptiveBoundaryReductionOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        initial_radius = 0.1  # Initial narrowing factor
        num_samples = max(1, int(5 * self.dim * (self.budget / (10 * self.dim))))
        initial_samples = self._uniform_sampling(bounds, num_samples)
        best_sample = None
        best_value = float('inf')
        improvement_threshold = 0.01 * (self.budget / 100)  # Adjust threshold based on budget

        for sample in initial_samples:
            if self.budget <= 0:
                break
            res = minimize(func, sample, method='L-BFGS-B', bounds=bounds)
            self.budget -= res.nfev
            if res.fun < best_value - improvement_threshold:
                best_value = res.fun
                best_sample = res.x
                # Reduce the search space around the best sample
                bounds = self._reduce_bounds(bounds, best_sample, initial_radius)
                initial_radius *= 0.9  # Gradually decrease radius for fine-tuning

        return best_sample

    def _uniform_sampling(self, bounds, num_samples):
        samples = []
        for _ in range(num_samples):
            sample = np.array([np.random.uniform(low, high) for low, high in bounds])
            samples.append(sample)
        return samples

    def _reduce_bounds(self, bounds, best_sample, radius):
        new_bounds = []
        for i, (low, high) in enumerate(bounds):
            center = best_sample[i]
            range_adjustment = (high - low) * radius
            new_low = max(low, center - range_adjustment)
            new_high = min(high, center + range_adjustment)
            new_bounds.append((new_low, new_high))
        return np.array(new_bounds)