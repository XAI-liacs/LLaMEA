import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        
        num_samples = min(self.dim * 2, max(3, self.budget // 5))
        
        samples = np.random.uniform(low=[b[0] for b in bounds],
                                    high=[b[1] for b in bounds],
                                    size=(num_samples, self.dim))
        
        evaluated_samples = [(x, func(x)) for x in samples]
        
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
        
        # Adjusted to resample after initial local optimization
        while remaining_budget > 0:
            result = minimize(count_calls, best_sample, method='L-BFGS-B', bounds=bounds, options={'maxiter': remaining_budget // 4})
            best_sample, best_cost = result.x, result.fun
            remaining_budget -= self.evaluation_count
            self.evaluation_count = 0

            if remaining_budget > 0:
                samples = np.random.uniform(low=[b[0] for b in bounds],
                                            high=[b[1] for b in bounds],
                                            size=(num_samples, self.dim))
                evaluated_samples = [(x, func(x)) for x in samples]
                evaluated_samples.sort(key=lambda item: item[1])
                best_sample, best_cost = evaluated_samples[0]
                remaining_budget -= num_samples

        return best_sample, best_cost