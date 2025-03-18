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
        
        weights = np.linspace(0.1, 1.0, initial_sample_size)
        weighted_samples = np.random.choice(initial_samples, size=initial_sample_size, p=weights/weights.sum())

        while evaluations < self.budget:
            local_bounds = [(max(lb[i], best_sample[i] - 5), min(ub[i], best_sample[i] + 5)) for i in range(self.dim)]
            res = minimize(func, best_sample, method='L-BFGS-B', bounds=local_bounds, options={'maxfun': self.budget - evaluations})
            momentum = 0.9 * (res.x - best_sample)
            best_sample = res.x + momentum
            best_score = res.fun
            evaluations += res.nfev

            lb = np.maximum(lb, best_sample - 0.1 * (ub - lb))
            ub = np.minimum(ub, best_sample + 0.1 * (ub - lb))
            
            if evaluations < self.budget // 2:
                initial_sample_size = max(10, self.budget // 20)
            else:
                initial_sample_size = max(10, self.budget // 15)

        return best_sample, best_score