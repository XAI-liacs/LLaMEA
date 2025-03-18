import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        bounds = func.bounds
        lb = bounds.lb
        ub = bounds.ub
        
        num_samples = min(max(5, self.budget // 5), 10)
        sampler = qmc.Sobol(d=self.dim, scramble=True if self.dim < 5 else False)
        samples = qmc.scale(sampler.random(n=num_samples**self.dim), lb, ub)
        
        best_sample = None
        best_value = float('inf')
        evaluations = 0
        
        # Sort samples based on variance of their coordinates to prioritize diverse regions
        samples = sorted(samples, key=lambda x: -np.var(x))
        
        for sample in samples:
            if evaluations >= self.budget:  # Ensure not to exceed budget
                break
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_sample = sample
        
        remaining_budget = self.budget - evaluations
        if remaining_budget > 0:
            result = minimize(func, best_sample, bounds=list(zip(lb, ub)), method='L-BFGS-B', options={'maxfun': remaining_budget, 'gtol': 1e-8, 'eps': 1e-6})
            if result.success and remaining_budget > result.nfev:
                result2 = minimize(func, result.x, bounds=list(zip(lb, ub)), method='Nelder-Mead', options={'maxfev': remaining_budget - result.nfev})
                return result2.x if result2.success else result.x
        return best_sample