import numpy as np
from scipy.optimize import minimize

class EnhancedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        initial_sample_count = max(10, self.budget // 8)

        initial_samples = []
        for _ in range(initial_sample_count):
            sample = np.array([
                np.random.uniform(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)
            ])
            initial_samples.append(sample)

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

        bounds = [(max(lb, x - 0.1 * (ub - lb)), min(ub, x + 0.1 * (ub - lb)))
                  for x, lb, ub in zip(best_sample, func.bounds.lb, func.bounds.ub)]

        def objective(x):
            return func(x)

        res1 = minimize(objective, x0=best_sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': int(self.budget * 0.4), 'ftol': 1e-8})
        
        if res1.success:
            best_sample = res1.x
            best_value = res1.fun
            self.budget -= res1.nfev

        bounds = [(max(lb, x - 0.05 * (ub - lb)), min(ub, x + 0.05 * (ub - lb))) for x, lb, ub in zip(best_sample, func.bounds.lb, func.bounds.ub)]

        res2 = minimize(objective, x0=best_sample, method='Nelder-Mead', bounds=bounds, options={'maxfev': int(self.budget * 0.6)})
        
        if res2.success and res2.fun < best_value:
            return res2.x
        else:
            return best_sample