import numpy as np
from scipy.optimize import minimize

class PhotonicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        
        num_initial_samples = min(self.budget // 3, 10)
        
        initial_samples = np.random.uniform(lb, ub, (num_initial_samples, self.dim))
        
        best_solution = None
        best_value = float('inf')
        
        for sample in initial_samples:
            result = minimize(func, sample, method='L-BFGS-B', bounds=np.array(list(zip(lb, ub))))
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
        
        remaining_budget = self.budget - num_initial_samples
        while remaining_budget > 0:
            if remaining_budget > self.budget // 4:
                current_bounds = [(max(lb[i], best_solution[i] - 0.2 * (ub[i] - lb[i])), min(ub[i], best_solution[i] + 0.2 * (ub[i] - lb[i]))) for i in range(self.dim)]
                result = minimize(func, best_solution, method='L-BFGS-B', bounds=current_bounds)
            else:
                result = minimize(func, best_solution, method='Nelder-Mead', options={'maxiter': remaining_budget})
            
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            
            remaining_budget -= 1
        
        return best_solution