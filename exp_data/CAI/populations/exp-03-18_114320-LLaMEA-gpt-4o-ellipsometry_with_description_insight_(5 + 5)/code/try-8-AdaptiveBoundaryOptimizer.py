import numpy as np
from scipy.optimize import minimize

class AdaptiveBoundaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        lb, ub = func.bounds.lb, func.bounds.ub

        # Calculate the number of initial samples with adaptive density
        initial_sample_count = min(max(10, self.budget // (2 * self.dim)), self.budget // 5)

        # Uniformly sample points within the bounds
        samples = [np.random.uniform(lb, ub) for _ in range(initial_sample_count)]

        # Evaluate initial samples and keep the best one
        sample_evals = [func(sample) for sample in samples]
        best_index = np.argmin(sample_evals)
        best_sample = samples[best_index]
        best_value = sample_evals[best_index]

        # Remaining budget after initial sampling
        remaining_budget = self.budget - initial_sample_count

        # Define the objective for BFGS
        def wrapped_func(x):
            nonlocal remaining_budget
            if remaining_budget <= 0:
                raise ValueError("Budget exceeded")
            remaining_budget -= 1
            return func(x)

        # Use BFGS with bounds to refine the best solution found
        result = minimize(wrapped_func, best_sample, method='L-BFGS-B', bounds=list(zip(lb, ub)))

        # Return the best solution found
        return result.x, result.fun