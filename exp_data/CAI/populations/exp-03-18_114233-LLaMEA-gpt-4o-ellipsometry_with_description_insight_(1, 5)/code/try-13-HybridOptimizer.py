import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        bounds = func.bounds
        lb = bounds.lb
        ub = bounds.ub
        
        num_samples = min(15, self.budget // 3)  # Adjusted sampling density
        grids = [np.linspace(lb[i], ub[i], num_samples) for i in range(self.dim)]
        samples = np.array(np.meshgrid(*grids)).T.reshape(-1, self.dim)
        
        best_sample = None
        best_value = float('inf')
        evaluations = 0
        
        for sample in samples:
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_sample = sample
        
        remaining_budget = self.budget - evaluations
        if remaining_budget > 0:
            result = minimize(func, best_sample, bounds=list(zip(lb, ub)), method='BFGS', options={'maxfun': remaining_budget})
            if result.success and remaining_budget > result.nfev:
                # Use another local method if budget allows
                result2 = minimize(func, result.x, bounds=list(zip(lb, ub)), method='Nelder-Mead', options={'maxfev': remaining_budget - result.nfev})
                return result2.x if result2.success else result.x
        return best_sample