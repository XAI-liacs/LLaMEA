import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        
        # Adaptive initial sampling based on budget and dimensionality
        num_samples = min(10, max(2, self.budget // (2 * self.dim)))  
        samples = np.random.uniform(low=[b[0] for b in bounds],
                                    high=[b[1] for b in bounds],
                                    size=(num_samples, self.dim))
        
        evaluated_samples = [(x, func(x)) for x in samples]
        
        # Sort samples based on their evaluated cost
        evaluated_samples.sort(key=lambda item: item[1])
        best_sample, best_cost = evaluated_samples[0]
        
        remaining_budget = self.budget - num_samples
        self.evaluation_count = 0
        
        def count_calls(x):
            if self.evaluation_count < remaining_budget:
                self.evaluation_count += 1
                return func(x)
            else:
                raise RuntimeError("Exceeded budget in local optimization")

        # Dual-phase optimization: initial with L-BFGS-B, fallback to Nelder-Mead if unsuccessful
        try:
            result = minimize(count_calls, best_sample, method='L-BFGS-B', bounds=bounds, options={'maxiter': remaining_budget // 2})
        except RuntimeError:
            result = minimize(count_calls, best_sample, method='Nelder-Mead', options={'maxiter': remaining_budget})
        
        return result.x, result.fun