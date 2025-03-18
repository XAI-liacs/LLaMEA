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
        
        # Initial exploration with improved sampling density
        num_samples = min(remaining_budget // 3, 20 * self.dim)  # Changed line: Increased exploration samples for better initial coverage
        samples = np.random.uniform(lb, ub, (num_samples, self.dim))
        remaining_budget -= num_samples
        
        # Evaluate samples and find the best initial guess
        sample_evaluations = np.array([func(sample) for sample in samples])
        best_idx = np.argmin(sample_evaluations)
        best_solution = samples[best_idx]
        best_value = sample_evaluations[best_idx]
        
        # Dynamic budget reshuffling for local search
        local_budget = int(max(10, remaining_budget * 0.9))  # Changed line: Allocate 90% of remaining budget for local optimization
        adaptive_maxfun = min(local_budget, 100)  # Changed line: Cap max function evaluations to prevent overshooting budget
        
        # Local optimization using BFGS with adaptive budget adjustment
        def local_objective(x):
            return func(x)
        
        result = minimize(local_objective, best_solution, bounds=list(zip(lb, ub)), method='L-BFGS-B', options={'maxfun': adaptive_maxfun})  # Changed lines: Use adaptive maxfun for budget adherence
        
        # Choose the best solution found
        if result.fun < best_value:
            best_solution = result.x
            best_value = result.fun
        
        return best_solution