import numpy as np
from scipy.optimize import minimize

class PhotonicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')
        
        # Calculate the budget for sampling and optimization
        sample_budget = max(1, self.budget // 10)
        optimize_budget = self.budget - sample_budget
        
        # Uniform sampling for initial guesses
        samples = np.random.uniform(lb, ub, (sample_budget, self.dim))
        
        for sample in samples:
            result = minimize(func, sample, method='Nelder-Mead', options={'maxiter': max(1, optimize_budget // sample_budget)})
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        return best_solution