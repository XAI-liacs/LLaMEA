import numpy as np
from scipy.optimize import minimize

class ABRLSOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Uniformly sample initial points within bounds
        n_initial_samples = min(5 * self.dim, self.budget // 2)
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        # Variance-based scaling for initial samples
        initial_samples = np.random.normal((bounds[:, 0] + bounds[:, 1]) / 2, 
                                           (bounds[:, 1] - bounds[:, 0]) / 4, 
                                           (n_initial_samples, self.dim))
        function_evals = 0
        
        # Evaluate initial samples
        best_value = float('inf')
        best_point = None
        for sample in initial_samples:
            value = func(sample)
            function_evals += 1
            if value < best_value:
                best_value = value
                best_point = sample

        # Start optimization from the best initial sample found
        x0 = best_point
        refined_bounds = bounds.copy()
        velocity = np.zeros(self.dim)  # Initialize velocity for momentum

        # Adaptive boundary refinement
        while function_evals < self.budget:
            def bounded_func(x):
                nonlocal function_evals
                if np.any(x < refined_bounds[:, 0]) or np.any(x > refined_bounds[:, 1]):
                    return float('inf')  # Penalty for out of bounds
                function_evals += 1
                return func(x)

            res = minimize(bounded_func, x0, method='L-BFGS-B', bounds=refined_bounds)
            if res.fun < best_value:
                best_value = res.fun
                best_point = res.x

            # Update the starting point and refine bounds
            x0 = res.x
            shrink_factor = 0.5 + 0.3 * (function_evals / self.budget)  # Dynamic shrink factor
            refined_bounds[:, 0] = np.maximum(refined_bounds[:, 0], x0 - (refined_bounds[:, 1] - refined_bounds[:, 0]) * (1 - shrink_factor) / 2)
            refined_bounds[:, 1] = np.minimum(refined_bounds[:, 1], x0 + (refined_bounds[:, 1] - refined_bounds[:, 0]) * (1 - shrink_factor) / 2)
            velocity = 0.9 * velocity + (refined_bounds[:, 1] - refined_bounds[:, 0]) * 0.1  # Momentum update

            if function_evals >= self.budget:
                break

        return best_point