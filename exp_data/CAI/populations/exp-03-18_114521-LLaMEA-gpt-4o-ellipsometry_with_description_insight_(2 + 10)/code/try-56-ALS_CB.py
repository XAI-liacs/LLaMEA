import numpy as np
from scipy.optimize import minimize

class ALS_CB:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Dynamic initial sampling size based on budget
        initial_sample_size = max(10, self.budget // 10)
        initial_samples = np.random.uniform(lb, ub, (initial_sample_size, self.dim))
        evaluated_samples = [(x, func(x)) for x in initial_samples]
        evaluated_samples.sort(key=lambda x: x[1])
        
        best_sample = evaluated_samples[0][0]
        best_score = evaluated_samples[0][1]

        evaluations = len(evaluated_samples)
        
        while evaluations < self.budget:
            # Optimize locally around the best solution found so far
            search_radius = 0.1 * (ub - lb)  # Adaptive search radius
            local_bounds = [(max(lb[i], best_sample[i] - search_radius[i]), min(ub[i], best_sample[i] + search_radius[i])) for i in range(self.dim)]
            res = minimize(func, best_sample, method='L-BFGS-B', bounds=local_bounds, options={'maxfun': self.budget - evaluations})
            
            best_sample = res.x
            best_score = res.fun
            evaluations += res.nfev

            # Restart mechanism if stuck in local optima
            if evaluations < self.budget and np.allclose(best_sample, evaluated_samples[0][0], atol=1e-2):
                best_sample = np.random.uniform(lb, ub, self.dim)
        
        return best_sample, best_score