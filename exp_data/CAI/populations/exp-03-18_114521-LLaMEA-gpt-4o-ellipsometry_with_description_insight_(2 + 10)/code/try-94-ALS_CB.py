import numpy as np
from scipy.optimize import minimize

class ALS_CB:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        initial_sample_size = max(10, self.budget // 10)
        initial_samples = np.random.uniform(lb, ub, (initial_sample_size, self.dim))
        evaluated_samples = [(x, func(x)) for x in initial_samples]
        evaluated_samples.sort(key=lambda x: x[1])
        
        best_sample = evaluated_samples[0][0]
        best_score = evaluated_samples[0][1]

        evaluations = len(evaluated_samples)
        
        while evaluations < self.budget:
            local_bounds = [(max(lb[i], best_sample[i] - np.random.uniform(1, 5)), 
                             min(ub[i], best_sample[i] + np.random.uniform(1, 5))) for i in range(self.dim)]
            res = minimize(func, best_sample, method='L-BFGS-B', bounds=local_bounds, 
                           options={'maxfun': self.budget - evaluations, 'ftol': 1e-9})
            
            if res.fun < best_score:
                best_sample = res.x
                best_score = res.fun
            evaluations += res.nfev

            lb = np.maximum(lb, best_sample - 0.1 * (ub - lb))
            ub = np.minimum(ub, best_sample + 0.1 * (ub - lb))
            
            initial_sample_size = max(10, self.budget // 20)

        return best_sample, best_score