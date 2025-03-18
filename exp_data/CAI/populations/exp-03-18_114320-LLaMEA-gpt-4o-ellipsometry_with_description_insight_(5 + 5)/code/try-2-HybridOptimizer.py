import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initial uniform sampling
        sample_size = min(self.budget // 5, 10)
        samples = np.random.uniform(lb, ub, (sample_size, self.dim))
        evaluations = [func(sample) for sample in samples]
        self.budget -= sample_size
        
        # Find the best initial sample
        best_sample_idx = np.argmin(evaluations)
        best_sample = samples[best_sample_idx]
        best_eval = evaluations[best_sample_idx]
        
        # Local optimization using BFGS
        def local_optimization(x):
            return func(x)
        
        opts = {'maxiter': self.budget, 'disp': False}
        result = minimize(local_optimization, best_sample, method='L-BFGS-B', bounds=list(zip(lb, ub)), options=opts)
        return result.x

# Example usage:
# optimizer = HybridOptimizer(budget=100, dim=2)
# best_solution = optimizer(func)