import numpy as np
from scipy.optimize import minimize, differential_evolution

class ALS_CB:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Change initial sampling to differential evolution
        strategy = differential_evolution(func, bounds=list(zip(lb, ub)), maxiter=1, popsize=max(10, self.budget // 10))
        initial_sample = strategy.x
        best_sample = initial_sample
        best_score = func(initial_sample)
        
        evaluations = strategy.nfev
        
        while evaluations < self.budget:
            # Optimize locally around the best solution found so far
            local_bounds = [(max(lb[i], best_sample[i] - 5), min(ub[i], best_sample[i] + 5)) for i in range(self.dim)]
            res = minimize(func, best_sample, method='L-BFGS-B', bounds=local_bounds, options={'maxfun': self.budget - evaluations})
            
            if res.fun < best_score:  # Update only if improved
                best_sample, best_score = res.x, res.fun
            
            evaluations += res.nfev

            # Adjust bounds and constraints iteratively for better exploration
            lb = np.maximum(lb, best_sample - 0.1 * (ub - lb))
            ub = np.minimum(ub, best_sample + 0.1 * (ub - lb))

        return best_sample, best_score