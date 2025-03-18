import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Calculate the number of initial samples
        initial_sample_count = max(10, self.budget // 10)
        
        # Randomly sample initial points within bounds using Sobol sequences
        sampler = Sobol(d=self.dim, scramble=True)
        initial_samples = sampler.random(n=initial_sample_count)
        initial_samples = initial_samples * (func.bounds.ub - func.bounds.lb) + func.bounds.lb

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
        bounds = [(max(lb, x - 0.1 * (ub - lb)), min(ub, x + 0.1 * (ub - lb)))
                  for x, lb, ub in zip(best_sample, func.bounds.lb, func.bounds.ub)]

        # Define the objective function for the local optimizer
        def objective(x):
            return func(x)

        # Use BFGS for local optimization within the new bounds
        res = minimize(objective, x0=best_sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget})

        return res.x