import numpy as np
from scipy.optimize import minimize

class ALS_CB:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.momentum = np.zeros(dim)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        initial_samples = np.random.uniform(lb, ub, (10, self.dim))
        evaluated_samples = [(x, func(x)) for x in initial_samples]
        evaluated_samples.sort(key=lambda x: x[1])
        
        best_sample = evaluated_samples[0][0]
        best_score = evaluated_samples[0][1]
        evaluations = len(evaluated_samples)
        
        while evaluations < self.budget:
            local_bounds = [(max(lb[i], best_sample[i] - 10), min(ub[i], best_sample[i] + 10)) for i in range(self.dim)]
            
            momentum_factor = 0.9
            self.momentum = momentum_factor * self.momentum + (1 - momentum_factor) * best_sample
            
            res = minimize(func, self.momentum, method='L-BFGS-B', bounds=local_bounds, options={'maxfun': self.budget - evaluations})
            
            if res.fun < best_score:
                best_sample, best_score = res.x, res.fun

            evaluations += res.nfev

            lb = np.maximum(lb, best_sample - 0.1 * (ub - lb))
            ub = np.minimum(ub, best_sample + 0.1 * (ub - lb))

        return best_sample, best_score