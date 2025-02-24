import numpy as np
from scipy.optimize import minimize

class PhotonicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        
        num_initial_samples = min(self.budget // 3, 15)  # Increased initial sampling for better exploration
        initial_samples = np.random.uniform(lb, ub, (num_initial_samples, self.dim))
        
        best_solution = None
        best_value = float('inf')
        
        for sample in initial_samples:
            adaptive_lr = 0.1 + 0.9 * np.random.rand()  # Adding adaptive learning rate for diversity
            result = minimize(func, sample, method='L-BFGS-B', 
                              bounds=np.array(list(zip(lb, ub))), options={'ftol': adaptive_lr})
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        remaining_budget = self.budget - num_initial_samples
        while remaining_budget > 0:
            current_bounds = [(max(lb[i], best_solution[i] - 0.15 * (ub[i] - lb[i])), 
                               min(ub[i], best_solution[i] + 0.15 * (ub[i] - lb[i]))) for i in range(self.dim)]
            
            result = minimize(func, best_solution, method='L-BFGS-B', bounds=current_bounds)
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            remaining_budget -= 1
        
        return best_solution