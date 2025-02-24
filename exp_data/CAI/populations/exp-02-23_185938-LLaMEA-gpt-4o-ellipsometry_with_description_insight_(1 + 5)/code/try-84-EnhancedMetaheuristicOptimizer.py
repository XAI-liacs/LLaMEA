import numpy as np
from scipy.optimize import minimize

class EnhancedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        initial_sample_count = max(15, self.budget // 5)  # Adjusted initial sample count for enhanced exploration
        
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

        # Narrow bounds around the best initial sample
        bounds = [(max(lb, x - 0.2 * (ub - lb)), min(ub, x + 0.2 * (ub - lb)))  # Adjust bounds more aggressively
                  for x, lb, ub in zip(best_sample, func.bounds.lb, func.bounds.ub)]

        # Define the objective function for the local optimizer
        def objective(x):
            return func(x)

        # Use dual-phase optimization: first L-BFGS-B then Nelder-Mead for fine-tuning
        res = minimize(objective, x0=best_sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': int(self.budget * 0.7), 'ftol': 1e-8})
        
        if res.success and self.budget > 0:
            # Further exploit with Nelder-Mead
            res_nm = minimize(objective, x0=res.x, method='Nelder-Mead', options={'maxiter': int(self.budget), 'xatol': 1e-8})
            return res_nm.x if res_nm.success else res.x
        else:
            return best_sample  # Fallback if optimization fails