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
        
        # Initial exploration with adaptive sampling based on variance
        num_samples = min(remaining_budget // 2, 15 * self.dim)
        samples = np.random.uniform(lb, ub, (num_samples, self.dim))
        remaining_budget -= num_samples
        
        # Evaluate samples and find the best initial guess
        sample_evaluations = np.array([func(sample) for sample in samples])
        best_idx = np.argmin(sample_evaluations)
        best_solution = samples[best_idx]
        best_value = sample_evaluations[best_idx]
        
        # Local optimization using Nelder-Mead
        def local_objective(x):
            return func(x)
        
        # Adjust budget for local optimization based on initial evaluations
        remaining_budget = max(remaining_budget, 5)
        result = minimize(local_objective, best_solution, bounds=list(zip(lb, ub)), method='Nelder-Mead', options={'maxfev': remaining_budget})
        
        # Choose the best solution found
        if result.fun < best_value:
            best_solution = result.x
            best_value = result.fun
        
        return best_solution