import numpy as np
from scipy.optimize import minimize, dual_annealing

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        remaining_budget = self.budget
        
        num_samples = min(remaining_budget // 4, 30 * self.dim)  # Adjust samples for dynamic exploration
        samples = np.random.uniform(lb, ub, (num_samples, self.dim))
        remaining_budget -= num_samples
        
        sample_evaluations = np.array([func(sample) for sample in samples])
        best_idx = np.argmin(sample_evaluations)
        best_solution = samples[best_idx]
        best_value = sample_evaluations[best_idx]
        
        def local_objective(x):
            return func(x)
        
        remaining_budget = max(remaining_budget, 5)
        
        # Choose method based on remaining budget
        if remaining_budget > 10:
            result = minimize(local_objective, best_solution, bounds=list(zip(lb, ub)), method='BFGS', options={'maxfev': remaining_budget // 2})
        else:
            result = dual_annealing(local_objective, bounds=list(zip(lb, ub)), maxfun=remaining_budget)
        
        if result.fun < best_value:
            best_solution = result.x
            best_value = result.fun
        
        return best_solution