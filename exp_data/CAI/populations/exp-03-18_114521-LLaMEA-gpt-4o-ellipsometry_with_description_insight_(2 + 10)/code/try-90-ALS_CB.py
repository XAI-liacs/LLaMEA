import numpy as np
from scipy.optimize import minimize

class ALS_CB:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        initial_sample_size = max(10, self.budget // 8)
        initial_samples = np.random.uniform(lb, ub, (initial_sample_size, self.dim))
        evaluated_samples = [(x, func(x)) for x in initial_samples]
        evaluated_samples.sort(key=lambda x: x[1])
        
        best_sample = evaluated_samples[0][0]
        best_score = evaluated_samples[0][1]

        evaluations = len(evaluated_samples)
        
        while evaluations < self.budget:
            local_bounds = [(max(lb[i], best_sample[i] - 3), min(ub[i], best_sample[i] + 3)) for i in range(self.dim)]
            res = minimize(func, best_sample, method='L-BFGS-B', bounds=local_bounds, options={'maxfun': self.budget - evaluations})
            
            new_sample = res.x
            new_score = res.fun
            evaluations += res.nfev
            
            if new_score < best_score:
                best_sample, best_score = new_sample, new_score
            elif np.random.rand() < 0.3:  # Resample strategy
                resample_size = min(5, self.budget - evaluations)
                resamples = np.random.uniform(lb, ub, (resample_size, self.dim))
                for sample in resamples:
                    score = func(sample)
                    evaluations += 1
                    if score < best_score:
                        best_sample, best_score = sample, score
                        break

            lb = np.maximum(lb, best_sample - 0.05 * (ub - lb))
            ub = np.minimum(ub, best_sample + 0.05 * (ub - lb))

        return best_sample, best_score