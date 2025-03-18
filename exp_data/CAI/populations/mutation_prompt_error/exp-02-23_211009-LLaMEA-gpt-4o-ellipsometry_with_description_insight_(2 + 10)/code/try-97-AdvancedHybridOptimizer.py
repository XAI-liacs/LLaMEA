import numpy as np
from scipy.optimize import minimize

class AdvancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Dynamically adjust initial sampling points based on budget and dimensionality
        initial_samples = max(min(self.budget // (2 * self.dim), 80), 15)  # Adjusted line
        remaining_budget = self.budget - initial_samples
        
        # Uniformly sample initial points with dynamic sampling strategy
        samples = np.random.uniform(lb, ub, (initial_samples, self.dim))
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
        
        # Use the remaining budget efficiently with adaptive L-BFGS-B enhanced by restarting from local minima
        options = {'maxiter': remaining_budget // 2, 'disp': False}  # Adjusted line
        result = minimize(bounded_func, best_solution, method='L-BFGS-B', bounds=np.array([lb, ub]).T, options=options)
        
        # Restart if budget allows for potential improvement
        if evaluations + result.nit < self.budget:  # Adjusted line
            remaining_budget = self.budget - evaluations - result.nit  # Adjusted line
            options['maxiter'] = remaining_budget
            # Re-optimize from current best solution
            result = minimize(bounded_func, result.x, method='L-BFGS-B', bounds=np.array([lb, ub]).T, options=options)  # Adjusted line
        
        return result.x