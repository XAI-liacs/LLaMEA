import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Initial uniform sampling in parameter space
        samples = self.initial_sampling(func)
        best_sample = min(samples, key=lambda x: x[1])
        
        # Adaptive boundary refinement
        bounds = self.adaptive_bounds(func.bounds, best_sample[0])

        # Local optimization using BFGS with dynamic step size reduction
        result = minimize(fun=func, x0=best_sample[0], bounds=bounds, method='L-BFGS-B', options={'maxfun': self.budget - self.evaluations, 'ftol': 1e-9})
        return result.x, result.fun

    def initial_sampling(self, func):
        num_initial_samples = min(10, self.budget // 2)
        lower_bounds = func.bounds.lb
        upper_bounds = func.bounds.ub
        samples = []
        
        for _ in range(num_initial_samples):
            sample = np.random.uniform(lower_bounds, upper_bounds, self.dim)
            value = func(sample)
            self.evaluations += 1
            samples.append((sample, value))
            if self.evaluations >= self.budget:
                break

        return samples

    def adaptive_bounds(self, bounds, best_sample):
        lb, ub = bounds.lb, bounds.ub
        adaptive_lb = np.maximum(lb, best_sample - (ub - lb) * 0.1)
        adaptive_ub = np.minimum(ub, best_sample + (ub - lb) * 0.1)
        return list(zip(adaptive_lb, adaptive_ub))