import numpy as np
from scipy.optimize import minimize

class AdaptiveBoundaryReductionOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.learning_rate = 0.05  # New learning rate for momentum

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        initial_radius = 0.1
        num_samples = max(1, int(5 * self.dim * (self.budget / (10 * self.dim))))
        initial_samples = self._uniform_sampling(bounds, num_samples)
        best_sample = None
        best_value = float('inf')
        velocity = np.zeros(self.dim)  # Initialize momentum
        improvement_threshold = 0.01
        no_improvement_steps = 0

        for sample in initial_samples:
            if self.budget <= 0:
                break
            res = minimize(func, sample, method='L-BFGS-B', bounds=bounds)
            self.budget -= res.nfev
            if res.fun < best_value - improvement_threshold:
                best_value = res.fun
                best_sample = res.x
                bounds = self._reduce_bounds(bounds, best_sample, initial_radius)
                initial_radius *= 0.9
                no_improvement_steps = 0
            else:
                no_improvement_steps += 1
                if no_improvement_steps > 5:
                    sample = self._momentum_restart(best_sample, velocity, bounds)
                    no_improvement_steps = 0
            initial_radius = max(0.01, initial_radius * (1.1 if no_improvement_steps > 3 else 0.9))
            velocity = self.learning_rate * (sample - best_sample)  # Update momentum

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

    def _momentum_restart(self, sample, velocity, bounds):
        new_sample = sample + velocity
        return np.clip(new_sample, bounds[:, 0], bounds[:, 1])  # Clip to bounds