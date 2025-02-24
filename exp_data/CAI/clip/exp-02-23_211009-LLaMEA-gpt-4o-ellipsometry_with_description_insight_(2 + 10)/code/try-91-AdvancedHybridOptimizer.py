import numpy as np
from scipy.optimize import minimize

class AdvancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Dynamically adjust initial sampling points based on budget and dimensionality
        initial_samples = max(min(self.budget // (3 * self.dim), 100), 10)
        remaining_budget = self.budget - initial_samples
        
        # Hybrid sampling strategy: Combine uniform and Gaussian sampling
        samples_uniform = np.random.uniform(lb, ub, (initial_samples // 2, self.dim))
        samples_gaussian = np.random.normal(loc=(lb + ub) / 2, scale=(ub - lb) / 6, size=(initial_samples // 2, self.dim))
        samples = np.vstack((samples_uniform, np.clip(samples_gaussian, lb, ub)))

        best_value = float('inf')
        best_solution = None
        
        # Evaluate sampled points
        evaluations = 0
        for sample in samples:
            if evaluations >= self.budget:
                break
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample
        
        # Define a bounded function to ensure the search remains within the specified bounds
        def bounded_func(x):
            return func(np.clip(x, lb, ub))
        
        # Adaptive local optimization with updated options
        options = {'maxiter': remaining_budget, 'disp': False}
        result = minimize(bounded_func, best_solution, method='L-BFGS-B', bounds=np.array([lb, ub]).T, options=options)
        
        return result.x