import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Step 1: Dynamic initial sampling based on remaining budget
        sample_size = min(20, self.budget // 3)  # Increase initial sample size
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        samples = np.random.uniform(bounds[:, 0], bounds[:, 1], (sample_size, self.dim))

        # Evaluate the initial samples
        sample_costs = [func(sample) for sample in samples]
        self.budget -= sample_size  # Update budget

        # Step 2: Multi-start local optimization
        num_starts = min(3, max(1, self.budget // 10))  # Dynamic number of local starts
        best_sample = samples[np.argmin(sample_costs)]
        best_cost = min(sample_costs)

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
        
        for _ in range(num_starts):  # Multi-start loop
            result = minimize(bounded_func, best_sample, method='L-BFGS-B', bounds=adaptive_bounds, options={'maxfun': self.budget // num_starts})
            if result.fun < best_cost:
                best_sample, best_cost = result.x, result.fun

        # Return the best-found solution
        return best_sample