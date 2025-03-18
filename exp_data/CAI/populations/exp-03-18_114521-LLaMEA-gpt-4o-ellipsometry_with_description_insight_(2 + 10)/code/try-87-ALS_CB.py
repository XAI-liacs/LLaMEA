import numpy as np
from scipy.optimize import minimize

class ALS_CB:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Hybrid initial sampling: uniform and Gaussian around the center
        center = (lb + ub) / 2
        initial_sample_size = max(10, self.budget // 10)
        uniform_samples = np.random.uniform(lb, ub, (initial_sample_size // 2, self.dim))
        gaussian_samples = np.random.normal(center, (ub - lb) / 6, (initial_sample_size - initial_sample_size // 2, self.dim))
        initial_samples = np.vstack((uniform_samples, gaussian_samples))
        initial_samples = np.clip(initial_samples, lb, ub)
        
        evaluated_samples = [(x, func(x)) for x in initial_samples]
        evaluated_samples.sort(key=lambda x: x[1])
        
        best_sample = evaluated_samples[0][0]
        best_score = evaluated_samples[0][1]

        evaluations = len(evaluated_samples)
        
        while evaluations < self.budget:
            # Adaptive local search: adjust optimization strategy based on evaluations left
            local_bounds = [(max(lb[i], best_sample[i] - 5), min(ub[i], best_sample[i] + 5)) for i in range(self.dim)]
            res = minimize(func, best_sample, method='L-BFGS-B', bounds=local_bounds, options={'maxfun': self.budget - evaluations})
            
            best_sample = res.x
            best_score = res.fun
            evaluations += res.nfev

            # Adjust bounds and constraints iteratively for better exploration
            lb = np.maximum(lb, best_sample - 0.1 * (ub - lb))
            ub = np.minimum(ub, best_sample + 0.1 * (ub - lb))

        return best_sample, best_score