import numpy as np
from scipy.optimize import minimize

class RefinedAdvancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Adaptively allocate initial samples based on budget and dimensionality, ensuring diversity
        initial_samples = max(min(self.budget // (4 * self.dim), 100), 15)
        remaining_budget = self.budget - initial_samples
        
        # Uniformly sample initial points with adaptive sample size for better exploration
        samples = np.random.uniform(lb, ub, (initial_samples, self.dim))
        best_value = float('inf')
        best_solution = None
        
        # Evaluate initial sampled points
        evaluations = 0
        for sample in samples:
            if evaluations >= self.budget:
                break
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample
        
        # Define a bounded function to ensure the search is within the specified region
        def bounded_func(x):
            return func(np.clip(x, lb, ub))
        
        # Refine search using L-BFGS-B starting from the best initial sample
        options = {'maxiter': min(remaining_budget, 50), 'disp': False}
        result = minimize(bounded_func, best_solution, method='L-BFGS-B', bounds=np.array([lb, ub]).T, options=options)
        
        # Further intensify search if budget allows
        if evaluations + options['maxiter'] < self.budget:
            result = minimize(bounded_func, result.x, method='L-BFGS-B', bounds=np.array([lb, ub]).T, options={'maxiter': self.budget - evaluations - options['maxiter'], 'disp': False})

        return result.x