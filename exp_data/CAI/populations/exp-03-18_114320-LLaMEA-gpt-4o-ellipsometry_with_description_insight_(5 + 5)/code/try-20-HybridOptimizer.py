import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        # Unpack bounds
        lb, ub = func.bounds.lb, func.bounds.ub
        # Remaining budget after initial exploration
        remaining_budget = self.budget
        
        # Initial exploration with uniform sampling
        num_samples = min(remaining_budget // 2, 12 * self.dim)  # Slightly increase samples
        samples = np.random.uniform(lb, ub, (num_samples, self.dim))
        remaining_budget -= num_samples
        
        # Evaluate samples and find the best initial guess
        sample_evaluations = np.array([func(sample) for sample in samples])
        best_idx = np.argmin(sample_evaluations)
        best_solution = samples[best_idx]
        best_value = sample_evaluations[best_idx]
        
        # Local optimization using BFGS
        def local_objective(x):
            return func(x)
        
        result = minimize(local_objective, best_solution, bounds=list(zip(lb, ub)), method='L-BFGS-B', options={'maxfun': max(remaining_budget, 1)})
        
        # Choose the best solution found
        if result.fun < best_value:
            best_solution = result.x
            best_value = result.fun
        
        # Restart strategy to enhance exploration
        additional_samples = np.random.uniform(lb, ub, (num_samples, self.dim))
        for sample in additional_samples:
            val = func(sample)
            if val < best_value:
                best_solution = sample
                best_value = val
            
        return best_solution