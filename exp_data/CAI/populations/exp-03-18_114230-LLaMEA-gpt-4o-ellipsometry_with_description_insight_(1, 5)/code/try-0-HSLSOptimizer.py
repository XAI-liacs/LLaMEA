import numpy as np
from scipy.optimize import minimize

class HSLSOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        
    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        best_x = None
        best_obj = float('inf')
        
        # Use a portion of the budget for initial sampling
        sample_budget = self.budget // 4
        samples = np.random.uniform(lb, ub, size=(sample_budget, self.dim))
        
        for x0 in samples:
            if self.calls >= self.budget:
                break
            
            res = minimize(self._wrapped_func(func), x0, method='L-BFGS-B', bounds=list(zip(lb, ub)))
            self.calls += res.nfev
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
            
            # Dynamically tighten bounds around current best solution
            lb = np.maximum(lb, best_x - 0.1 * (ub - lb))
            ub = np.minimum(ub, best_x + 0.1 * (ub - lb))
        
        return best_x
    
    def _wrapped_func(self, func):
        def wrapper(x):
            if self.calls < self.budget:
                self.calls += 1
                return func(x)
            else:
                raise Exception("Budget exhausted")
        return wrapper