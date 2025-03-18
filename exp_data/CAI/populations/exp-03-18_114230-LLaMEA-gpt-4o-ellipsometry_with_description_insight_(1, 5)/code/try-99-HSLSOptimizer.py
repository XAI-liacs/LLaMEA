import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class HSLSOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        
    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        best_x = None
        best_obj = float('inf')
        
        sample_budget = self.budget // 3 
        sobol_sampler = Sobol(d=self.dim, scramble=True)
        samples = sobol_sampler.random_base2(m=int(np.log2(sample_budget)))  # Changed line

        samples = lb + samples * (ub - lb)
        
        for x0 in samples:
            if self.calls >= self.budget:
                break
            
            res = minimize(self._wrapped_func(func), x0, method='L-BFGS-B', bounds=list(zip(lb, ub)))
            self.calls += res.nfev
            if res.success and res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x

            contraction_factor = 0.02 * (self.budget - self.calls) / self.budget  # Changed line
            if best_x is not None:
                lb = np.maximum(lb, best_x - contraction_factor * (ub - lb))  
                ub = np.minimum(ub, best_x + contraction_factor * (ub - lb))
            if self.calls < self.budget:
                sample_budget = min(max(self.budget // 5, 1), self.budget - self.calls)  
                samples = sobol_sampler.random(sample_budget)
                samples = lb + samples * (ub - lb)
        
        return best_x
    
    def _wrapped_func(self, func):
        def wrapper(x):
            if self.calls < self.budget:
                self.calls += 1
                return func(x)
            else:
                return float('inf')
        return wrapper