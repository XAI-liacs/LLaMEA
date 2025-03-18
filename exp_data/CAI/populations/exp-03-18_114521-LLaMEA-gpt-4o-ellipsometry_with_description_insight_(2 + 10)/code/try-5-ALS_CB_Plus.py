import numpy as np
from scipy.optimize import minimize

class ALS_CB_Plus:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        initial_samples = np.random.uniform(lb, ub, (10, self.dim))
        evaluated_samples = [(x, func(x)) for x in initial_samples]
        evaluated_samples.sort(key=lambda x: x[1])
        
        best_sample = evaluated_samples[0][0]
        best_score = evaluated_samples[0][1]

        evaluations = len(evaluated_samples)
        cooling_rate = 0.95  # Simulated Annealing Cooling Rate
        
        while evaluations < self.budget:
            temperature = max(1.0, cooling_rate ** (evaluations // 5))
            local_bounds = [(max(lb[i], best_sample[i] - temperature), min(ub[i], best_sample[i] + temperature)) for i in range(self.dim)]
            res = minimize(func, best_sample, method='L-BFGS-B', bounds=local_bounds, options={'maxfun': self.budget - evaluations})
            
            if res.fun < best_score:
                best_sample = res.x
                best_score = res.fun
            
            evaluations += res.nfev

            lb = np.maximum(lb, best_sample - 0.05 * (ub - lb))
            ub = np.minimum(ub, best_sample + 0.05 * (ub - lb))

        return best_sample, best_score