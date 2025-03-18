import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        # Unpack bounds
        lb, ub = func.bounds.lb, func.bounds.ub
        remaining_budget = self.budget
        
        # Initial exploration with diversified sampling
        num_samples = min(remaining_budget // 3, 25 * self.dim)  # Increase sampling for better initial exploration
        samples = np.random.uniform(lb, ub, (num_samples, self.dim))
        remaining_budget -= num_samples
        
        # Evaluate samples and find the best initial guess
        sample_evaluations = np.array([func(sample) for sample in samples])
        best_idx = np.argmin(sample_evaluations)
        best_solution = samples[best_idx]
        best_value = sample_evaluations[best_idx]
        
        # Local optimization using adaptive method selection
        def local_objective(x):
            return func(x)
        
        # Adjust budget and choose method adaptively
        if remaining_budget > 10:
            method = 'L-BFGS-B'
        else:
            method = 'Nelder-Mead'
        
        result = minimize(local_objective, best_solution, bounds=list(zip(lb, ub)), method=method, options={'maxfun': remaining_budget})
        
        # Choose the best solution found
        if result.fun < best_value:
            best_solution = result.x
            best_value = result.fun
        
        return best_solution