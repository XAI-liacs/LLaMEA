import numpy as np
from scipy.optimize import minimize

class AdaptiveLocalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Define search bounds
        lb, ub = func.bounds.lb, func.bounds.ub

        # Hybrid sampling strategy: Combine uniform and Latin Hypercube Sampling
        init_points = np.vstack([np.random.uniform(lb, ub, size=(10, self.dim)),  # From 15 to 10 uniform samples
                                self.latin_hypercube_sampling(lb, ub, 5)])  # Added Latin Hypercube Sampling

        best_solution = None
        best_value = float('inf')

        for point in init_points:
            if self.evaluations >= self.budget:
                break

            # Optimize using local optimizer starting from the initial point
            result = minimize(self.bounded_func(func, lb, ub), point, method='Nelder-Mead',
                              options={'maxfev': self.budget - self.evaluations})

            # Count the number of function evaluations
            self.evaluations += result.nfev

            # Update best solution if found
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            # Adjust the search bounds adaptively and dynamically around best known solution
            lb = np.maximum(lb, best_solution - 0.1 * (ub - lb))  # Changed from 0.05 to 0.1
            ub = np.minimum(ub, best_solution + 0.1 * (ub - lb))  # Changed from 0.05 to 0.1

        return best_solution

    def bounded_func(self, func, lb, ub):
        def func_with_bounds(x):
            # Clip the solution to remain within bounds
            x_clipped = np.clip(x, lb, ub)
            return func(x_clipped)
        return func_with_bounds

    def latin_hypercube_sampling(self, lb, ub, n_samples):
        """Generate samples using Latin Hypercube Sampling."""
        l_bounds = np.array(lb)
        u_bounds = np.array(ub)
        dim = len(lb)
        result = np.zeros((n_samples, dim))
        for i in range(dim):
            # Using Latin Hypercube Sampling for enhanced space-filling
            result[:, i] = np.random.permutation(n_samples) / n_samples
        return l_bounds + (u_bounds - l_bounds) * result