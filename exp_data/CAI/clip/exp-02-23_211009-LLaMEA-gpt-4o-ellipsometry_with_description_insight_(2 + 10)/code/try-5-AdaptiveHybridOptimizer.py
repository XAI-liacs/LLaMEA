import numpy as np
from scipy.optimize import minimize

class AdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        initial_samples = min(self.budget // 10, 100)
        remaining_budget = self.budget - initial_samples
        
        samples = np.random.uniform(lb, ub, (initial_samples, self.dim))
        best_value = float('inf')
        best_solution = None
        
        evaluations = 0
        for sample in samples:
            if evaluations >= self.budget:
                break
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample
        
        def bounded_func(x):
            return func(np.clip(x, lb, ub))
        
        local_budget = remaining_budget // 2
        options = {'maxiter': local_budget, 'disp': False}
        
        # Start with BFGS
        result_bfgs = minimize(bounded_func, best_solution, method='L-BFGS-B', bounds=np.array([lb, ub]).T, options=options)
        if result_bfgs.fun < best_value:
            best_value = result_bfgs.fun
            best_solution = result_bfgs.x
        
        # Use the remaining budget for Nelder-Mead
        local_budget = remaining_budget - local_budget
        options = {'maxiter': local_budget, 'disp': False}
        
        result_nm = minimize(bounded_func, best_solution, method='Nelder-Mead', options=options)
        if result_nm.fun < best_value:
            best_value = result_nm.fun
            best_solution = result_nm.x
        
        return best_solution