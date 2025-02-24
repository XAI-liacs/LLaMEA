import numpy as np
from scipy.optimize import minimize

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Dynamically adjust initial sampling points based on budget and dimensionality
        initial_samples = max(min(self.budget // (3 * self.dim), 100), 10)  
        remaining_budget = self.budget - initial_samples
        
        # Use a Sobol sequence for better initial sampling distribution
        samples = sobol_sequence(lb, ub, initial_samples, self.dim)  # Changed line
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
        
        # Use multi-start local optimization with adaptive L-BFGS-B
        starts = 3  # Additional starting points for local search, Changed line
        options = {'maxiter': remaining_budget // starts, 'disp': False}  # Adjusted line
        for _ in range(starts):  # Changed line
            result = minimize(bounded_func, best_solution, method='L-BFGS-B', 
                              bounds=np.array([lb, ub]).T, options=options)
            if result.fun < best_value:  # Changed line
                best_value = result.fun  # Changed line
                best_solution = result.x  # Changed line
        
        return best_solution

def sobol_sequence(lb, ub, n, dim):  # New function
    from scipy.stats.qmc import Sobol
    sampler = Sobol(d=dim, scramble=False)
    sample = sampler.random(n)
    return lb + (ub - lb) * sample