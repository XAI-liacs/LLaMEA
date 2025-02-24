import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class EnhancedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        initial_sample_count = max(15, self.budget // 5)  # Adjust sample count for better initial approximation

        # Sobol sequence for quasi-random initial points within bounds
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        initial_points = sampler.random_base2(m=int(np.log2(initial_sample_count)))
        initial_samples = [func.bounds.lb + (func.bounds.ub - func.bounds.lb) * point for point in initial_points]

        # Evaluate initial samples and find the best one
        best_sample = None
        best_value = float('inf')
        for sample in initial_samples:
            if self.budget <= 0:
                return best_sample
            value = func(sample)
            self.budget -= 1
            if value < best_value:
                best_value = value
                best_sample = sample

        # Narrow bounds around the best initial sample
        bounds = [(max(lb, x - 0.2 * (ub - lb)), min(ub, x + 0.2 * (ub - lb)))
                  for x, lb, ub in zip(best_sample, func.bounds.lb, func.bounds.ub)]

        # Define the objective function for the local optimizer
        def objective(x):
            return func(x)

        # Hybrid optimization: try Nelder-Mead first, then L-BFGS-B
        res = minimize(objective, x0=best_sample, method='Nelder-Mead', options={'maxfev': int(self.budget * 0.3), 'xatol': 1e-8})
        if not res.success:
            res = minimize(objective, x0=res.x, method='L-BFGS-B', bounds=bounds, options={'maxfun': int(self.budget * 0.6), 'ftol': 1e-8})

        if res.success:
            return res.x
        else:
            return best_sample  # Fallback if optimization fails