import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class EnhancedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        initial_sample_count = max(12, self.budget // 6)  # Increase initial sample count for better exploration
        
        # Generate initial points using Sobol sequence for better exploration
        sampler = Sobol(d=self.dim, scramble=True)
        sobol_samples = sampler.random_base2(m=int(np.log2(initial_sample_count)))
        initial_samples = [
            func.bounds.lb + sample * (func.bounds.ub - func.bounds.lb)
            for sample in sobol_samples
        ]

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
        bounds = [(max(lb, x - 0.1 * (ub - lb)), min(ub, x + 0.1 * (ub - lb)))  # Adjust bounds less aggressively
                  for x, lb, ub in zip(best_sample, func.bounds.lb, func.bounds.ub)]

        # Define the objective function for the local optimizer
        def objective(x):
            return func(x)

        # Use L-BFGS-B for local optimization with adaptive options
        res = minimize(objective, x0=best_sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': int(self.budget * 0.9), 'ftol': 1e-8})  # Adjusted budget and tolerance

        if res.success:
            return res.x
        else:
            return best_sample  # Fallback if optimization fails