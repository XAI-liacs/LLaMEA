import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        sample_size = max(5, int(self.budget // 3))  # Adjusted sample size for better initial exploration
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        samples = np.random.uniform(bounds[:, 0], bounds[:, 1], (sample_size, self.dim))
        
        sample_costs = [func(sample) for sample in samples]
        self.budget -= sample_size

        best_index = np.argmin(sample_costs)
        best_sample = samples[best_index]

        adaptive_bounds = bounds.copy()
        for i in range(self.dim):
            span = (adaptive_bounds[i, 1] - adaptive_bounds[i, 0]) * 0.15  # 15% of the range
            adaptive_bounds[i, 0] = max(func.bounds.lb[i], best_sample[i] - span)
            adaptive_bounds[i, 1] = min(func.bounds.ub[i], best_sample[i] + span)

        def bounded_func(x):
            if np.all(x >= func.bounds.lb) and np.all(x <= func.bounds.ub):
                return func(x)
            return np.inf

        # Enhanced local optimization using Nelder-Mead within adjusted bounds
        result = minimize(bounded_func, best_sample, method='Nelder-Mead', options={'maxfev': self.budget})

        return result.x if result.success else best_sample