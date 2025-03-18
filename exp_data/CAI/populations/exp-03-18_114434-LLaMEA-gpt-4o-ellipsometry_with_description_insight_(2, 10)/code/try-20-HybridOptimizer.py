import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Step 1: Initial uniform sampling
        sample_size = min(10, self.budget // 2)  # Use a small portion of the budget for sampling
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        
        # Use Latin Hypercube Sampling instead of uniform sampling
        sampler = qmc.LatinHypercube(d=self.dim)
        samples = qmc.scale(sampler.random(sample_size), func.bounds.lb, func.bounds.ub)
        
        # Evaluate the initial samples
        sample_costs = [func(sample) for sample in samples]
        self.budget -= sample_size  # Update budget

        # Step 2: Select the best initial sample as starting point for local optimization
        best_index = np.argmin(sample_costs)
        best_sample = samples[best_index]
        best_cost = sample_costs[best_index]

        # Step 3: Adaptive bounds adjustment
        adaptive_bounds = bounds.copy()
        for i in range(self.dim):
            span = (adaptive_bounds[i, 1] - adaptive_bounds[i, 0]) * 0.1  # 10% of the range
            adaptive_bounds[i, 0] = max(func.bounds.lb[i], best_sample[i] - span)
            adaptive_bounds[i, 1] = min(func.bounds.ub[i], best_sample[i] + span)

        # Step 4: Local optimization using TNC within adjusted bounds
        def bounded_func(x):
            if np.all(x >= func.bounds.lb) and np.all(x <= func.bounds.ub):
                return func(x)
            return np.inf

        result = minimize(bounded_func, best_sample, method='TNC', bounds=adaptive_bounds, options={'maxfun': self.budget})
        
        # Return the best-found solution
        return result.x if result.success else best_sample