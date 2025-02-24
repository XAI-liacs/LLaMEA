import numpy as np
from scipy.optimize import minimize

class RefinedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Calculate the number of initial samples
        initial_sample_count = max(10, self.budget // 10)
        
        # Randomly sample initial points within bounds
        initial_samples = []
        for _ in range(initial_sample_count):
            sample = np.array([
                np.random.uniform(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)
            ])
            initial_samples.append(sample)

        # Evaluate initial samples and find the best one
        best_sample = None
        best_value = float('inf')
        for sample in initial_samples:
            value = func(sample)
            self.budget -= 1
            if value < best_value:
                best_value = value
                best_sample = sample
            if self.budget <= 0:
                return best_sample

        # Adaptive sampling: refine bounds around the best sample
        reduction_factor = 0.1
        bounds = [(max(lb, x - reduction_factor * (ub - lb)), min(ub, x + reduction_factor * (ub - lb)))
                  for x, lb, ub in zip(best_sample, func.bounds.lb, func.bounds.ub)]

        # Define the objective function for the local optimizer
        def objective(x):
            return func(x)

        # Dual local optimization: BFGS followed by Nelder-Mead
        res_bfgs = minimize(objective, x0=best_sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget // 2})
        self.budget -= res_bfgs.nfev

        if self.budget > 0:
            res_nelder = minimize(objective, x0=res_bfgs.x, method='Nelder-Mead', options={'maxfev': self.budget})
            self.budget -= res_nelder.nfev
            final_result = res_nelder.x
        else:
            final_result = res_bfgs.x

        return final_result