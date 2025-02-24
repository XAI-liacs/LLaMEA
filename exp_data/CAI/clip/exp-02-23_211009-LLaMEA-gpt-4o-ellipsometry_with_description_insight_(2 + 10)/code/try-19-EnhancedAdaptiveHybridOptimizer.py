import numpy as np
from scipy.optimize import minimize

class EnhancedAdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Dynamically adjust initial sampling points based on budget and dimensionality
        initial_samples = max(min(self.budget // (2 * self.dim), 100), 10)
        remaining_budget = self.budget - initial_samples
        
        # Uniformly sample initial points with probabilistic sampling strategy
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

        # Probabilistically decide whether to start local search from best or random sample
        if np.random.rand() < 0.5 and remaining_budget > 0:
            random_sample = samples[np.random.randint(0, initial_samples)]
            bounded_func = lambda x: func(np.clip(x, lb, ub))
            options = {'maxiter': remaining_budget // 2, 'disp': False}
            result = minimize(bounded_func, random_sample, method='L-BFGS-B', bounds=np.array([lb, ub]).T, options=options)
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            remaining_budget -= options['maxiter']

        # Define a bounded function to ensure the search remains within the specified bounds
        def bounded_func(x):
            return func(np.clip(x, lb, ub))
        
        # Use the remaining budget efficiently in local optimization with adaptive L-BFGS-B
        options = {'maxiter': remaining_budget, 'disp': False}
        result = minimize(bounded_func, best_solution, method='L-BFGS-B', bounds=np.array([lb, ub]).T, options=options)
        
        return result.x