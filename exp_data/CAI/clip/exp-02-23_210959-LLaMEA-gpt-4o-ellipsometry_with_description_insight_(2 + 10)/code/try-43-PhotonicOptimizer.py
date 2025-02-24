import numpy as np
from scipy.optimize import minimize

class PhotonicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        
        num_initial_samples = min(int(0.4 * self.budget), max(10, self.dim))
        
        initial_samples = np.random.uniform(lb, ub, (num_initial_samples, self.dim))
        
        best_solution = None
        best_value = float('inf')
        
        for sample in initial_samples:
            result = minimize(func, sample, method='BFGS', bounds=np.array(list(zip(lb, ub))))
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
        
        remaining_budget = self.budget - num_initial_samples
        
        while remaining_budget > 0:
            adaptive_bounds = [(max(lb[i], best_solution[i] - 0.15 * (ub[i] - lb[i])), min(ub[i], best_solution[i] + 0.15 * (ub[i] - lb[i]))) for i in range(self.dim)]
            
            perturbed_initial = best_solution + np.random.normal(0, 0.05, self.dim)
            perturbed_initial = np.clip(perturbed_initial, lb, ub)
            
            result = minimize(func, perturbed_initial, method='BFGS', bounds=adaptive_bounds)
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            
            remaining_budget -= 1
        
        return best_solution