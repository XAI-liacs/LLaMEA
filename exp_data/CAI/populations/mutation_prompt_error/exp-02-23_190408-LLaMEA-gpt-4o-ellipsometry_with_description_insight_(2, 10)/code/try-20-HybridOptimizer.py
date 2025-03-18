import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        num_initial_samples = min(5 * self.dim, self.budget // 2)
        samples = self.uniform_sampling(bounds, num_initial_samples)

        best_sample = None
        best_value = float('inf')
        
        for sample in samples:
            value = func(sample)
            if value < best_value:
                best_value = value
                best_sample = sample
        
        # Strategic re-sampling based on initial variance
        remaining_budget = self.budget - num_initial_samples
        if remaining_budget > 0:
            additional_samples = self.variance_based_sampling(samples, bounds)
            for sample in additional_samples:
                value = func(sample)
                remaining_budget -= 1
                if value < best_value:
                    best_value = value
                    best_sample = sample
        
        res = minimize(func, best_sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': remaining_budget})
        
        return res.x, res.fun

    def uniform_sampling(self, bounds, num_samples):
        samples = []
        for _ in range(num_samples):
            sample = [np.random.uniform(low, high) for low, high in bounds]
            samples.append(sample)
        return samples
    
    def variance_based_sampling(self, samples, bounds):
        variances = np.var(samples, axis=0)
        additional_samples = []
        for _ in range(3):  # Add three more strategic samples
            sample = [np.random.uniform(low, high) if var < 0.5 else np.mean([low, high]) for var, (low, high) in zip(variances, bounds)]
            additional_samples.append(sample)
        return additional_samples