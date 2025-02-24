import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Uniformly sample initial points
        num_initial_samples = min(10, self.budget // 2)
        initial_samples = [np.random.uniform(low=func.bounds.lb, high=func.bounds.ub)
                           for _ in range(num_initial_samples)]
        
        best_sample = None
        best_value = float('inf')
        
        # Evaluate initial samples
        for sample in initial_samples:
            if self.evaluations >= self.budget:
                break
            value = func(sample)
            self.evaluations += 1
            if value < best_value:
                best_value = value
                best_sample = sample

        # If budget allows, perform local optimization
        if self.evaluations < self.budget:
            result = minimize(func, x0=best_sample, method='L-BFGS-B',
                              bounds=list(zip(func.bounds.lb, func.bounds.ub)),
                              options={'maxfun': self.budget - self.evaluations})
            self.evaluations += result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_sample = result.x

        return best_sample, best_value