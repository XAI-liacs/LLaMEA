import numpy as np
from scipy.optimize import minimize

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Get bounds from the function
        lb, ub = func.bounds.lb, func.bounds.ub

        # Initial sampling strategy
        initial_samples = min(self.budget // 20, 50)
        adaptive_samples = max(self.budget // 5, 10)
        remaining_budget = self.budget - initial_samples - adaptive_samples

        # Uniformly sample initial points
        samples = np.random.uniform(lb, ub, (initial_samples, self.dim))
        best_value = float('inf')
        best_solution = None
        evaluations = 0

        # Evaluate initial sampled points
        for sample in samples:
            if evaluations >= self.budget:
                break
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample

        # Adaptive sampling based on current best solution
        for _ in range(adaptive_samples):
            if evaluations >= self.budget:
                break
            # Generate samples around the best solution
            new_sample = np.random.uniform(
                np.maximum(lb, best_solution - 0.1 * (ub - lb)),
                np.minimum(ub, best_solution + 0.1 * (ub - lb)),
                self.dim
            )
            value = func(new_sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = new_sample

        # Use local optimization with BFGS starting from the best evaluated point
        def bounded_func(x):
            return func(np.clip(x, lb, ub))

        # Allocate remaining budget for local optimization
        options = {'maxiter': remaining_budget, 'disp': False}
        result = minimize(bounded_func, best_solution, method='L-BFGS-B', bounds=np.array([lb, ub]).T, options=options)

        return result.x