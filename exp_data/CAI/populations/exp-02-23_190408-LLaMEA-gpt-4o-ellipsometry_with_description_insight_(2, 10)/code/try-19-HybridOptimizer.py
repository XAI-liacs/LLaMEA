import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        num_initial_samples = min(5 * self.dim, self.budget // 2)
        samples = self.adaptive_sampling(func, bounds, num_initial_samples)
        
        best_sample = None
        best_value = float('inf')
        
        for sample in samples:
            value = func(sample)
            if value < best_value:
                best_value = value
                best_sample = sample
        
        remaining_budget = self.budget - num_initial_samples
        
        res = minimize(func, best_sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': remaining_budget})
        
        return res.x, res.fun

    def adaptive_sampling(self, func, bounds, num_samples):
        samples = []
        variances = [(high - low) for low, high in bounds]
        for _ in range(num_samples):
            sample = [np.random.normal(loc=(low + high) / 2, scale=var/4) for (low, high), var in zip(bounds, variances)]
            samples.append(np.clip(sample, [low for low, _ in bounds], [high for _, high in bounds]))
        return samples