import numpy as np
from scipy.optimize import minimize

class EnhancedHybridOptimizer:
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

        remaining_budget = self.budget - num_initial_samples
        
        res = self.local_optimization(func, best_sample, bounds, remaining_budget)
        
        return res.x, res.fun

    def uniform_sampling(self, bounds, num_samples):
        samples = []
        for _ in range(num_samples):
            sample = np.random.uniform([low for low, _ in bounds],
                                       [high for _, high in bounds])
            samples.append(sample)
        return samples

    def local_optimization(self, func, initial_guess, bounds, budget):
        res = minimize(func, initial_guess, method='L-BFGS-B', bounds=bounds, options={'maxfun': budget, 'ftol': 1e-9})
        return res