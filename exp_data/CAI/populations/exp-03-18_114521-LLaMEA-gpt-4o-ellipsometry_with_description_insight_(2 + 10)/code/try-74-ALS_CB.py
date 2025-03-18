import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class ALS_CB:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Dynamic initial sampling size based on budget
        initial_sample_size = max(10, self.budget // 10)
        sobol = Sobol(d=self.dim, scramble=True)
        initial_samples = sobol.random(n=initial_sample_size) * (ub - lb) + lb
        evaluated_samples = [(x, func(x)) for x in initial_samples]
        evaluated_samples.sort(key=lambda x: x[1])
        
        best_sample = evaluated_samples[0][0]
        best_score = evaluated_samples[0][1]

        evaluations = len(evaluated_samples)
        
        while evaluations < self.budget:
            # Optimize locally around the best solution found so far
            local_bounds = [(max(lb[i], best_sample[i] - 5), min(ub[i], best_sample[i] + 5)) for i in range(self.dim)]
            res = minimize(func, best_sample, method='L-BFGS-B', bounds=local_bounds, options={'maxfun': self.budget - evaluations})
            
            best_sample = res.x
            best_score = res.fun
            evaluations += res.nfev

            # Adjust bounds and constraints iteratively for better exploration
            lb = np.maximum(lb, best_sample - 0.1 * (ub - lb))
            ub = np.minimum(ub, best_sample + 0.1 * (ub - lb))

        return best_sample, best_score