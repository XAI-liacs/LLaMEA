import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Step 1: Initial uniform sampling
        sample_size = min(15, self.budget // 2)  # Use a slightly larger portion of the budget for sampling
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        samples = np.random.uniform(bounds[:, 0], bounds[:, 1], (sample_size, self.dim))
        
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
            span = (adaptive_bounds[i, 1] - adaptive_bounds[i, 0]) * 0.05  # 5% of the range
            adaptive_bounds[i, 0] = max(func.bounds.lb[i], best_sample[i] - span)
            adaptive_bounds[i, 1] = min(func.bounds.ub[i], best_sample[i] + span)

        # Step 4: Local optimization using BFGS within adjusted bounds
        def bounded_func(x):
            if np.all(x >= func.bounds.lb) and np.all(x <= func.bounds.ub):
                return func(x)
            return np.inf

        # Modified line: Attempt a random restart if the solution is not successful
        result = minimize(bounded_func, best_sample, method='L-BFGS-B', bounds=adaptive_bounds, options={'maxfun': self.budget // 2})
        if not result.success:
            restart_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], self.dim)
            result = minimize(bounded_func, restart_sample, method='L-BFGS-B', bounds=adaptive_bounds, options={'maxfun': self.budget // 2})

        # Return the best-found solution
        return result.x if result.success else best_sample