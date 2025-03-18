import numpy as np
from scipy.optimize import minimize

class AdvancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        num_initial_samples = min(5 * self.dim, max(self.budget // 3, 1))
        samples = self.uniform_sampling(bounds, num_initial_samples)
        
        best_sample = None
        best_value = float('inf')
        
        for sample in samples:
            value = func(sample)
            if value < best_value:
                best_value = value
                best_sample = sample

        remaining_budget = self.budget - num_initial_samples
        res = self.adaptive_local_optimization(func, best_sample, bounds, remaining_budget)
        
        return res.x, res.fun

    def uniform_sampling(self, bounds, num_samples):
        return [np.random.uniform([low for low, _ in bounds], 
                                  [high for _, high in bounds]) for _ in range(num_samples)]

    def adaptive_local_optimization(self, func, initial_guess, bounds, budget):
        callback_info = {'func_calls': 0, 'best_value': float('inf')}

        def callback(xk):
            value = func(xk)
            callback_info['func_calls'] += 1
            if value < callback_info['best_value']:
                callback_info['best_value'] = value

            # Early stopping if improvement is very small
            if callback_info['func_calls'] > 5 and abs(callback_info['best_value']) < 1e-6:
                return True

        res = minimize(func, initial_guess, method='L-BFGS-B', bounds=bounds, 
                       callback=callback, options={'maxfun': budget, 'disp': False})
        
        if callback_info['func_calls'] < budget * 0.8:  # If budget not exhausted, resample
            new_samples = self.uniform_sampling(bounds, int((budget - callback_info['func_calls']) * 0.5))
            for sample in new_samples:
                resample_value = func(sample)
                if resample_value < res.fun:
                    res = minimize(func, sample, method='L-BFGS-B', bounds=bounds, 
                                   options={'maxfun': budget - callback_info['func_calls']})
                    if resample_value < res.fun:
                        res.fun = resample_value
                        res.x = sample

        return res