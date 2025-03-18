import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Ensure initial budget usage for a global search
        initial_samples = int(self.budget * 0.2)  # Adjusted initial sample percentage
        remaining_budget = self.budget - initial_samples

        # Uniform sampling for initial guesses
        lb, ub = func.bounds.lb, func.bounds.ub
        samples = np.random.uniform(lb, ub, (initial_samples, self.dim))
        best_sample, best_value = None, float('inf')

        for sample in samples:
            value = func(sample)
            if value < best_value:
                best_value = value
                best_sample = sample

        # Local optimization using Trust-Region Reflective algorithm
        bounds = [(lb[i], ub[i]) for i in range(self.dim)]
        remaining_evaluations = remaining_budget // self.dim
        
        # Use numerical gradient approximation for better refinement
        def wrapped_func(x):
            nonlocal remaining_evaluations
            if remaining_evaluations <= 0:
                raise StopIteration("Budget exceeded")
            remaining_evaluations -= 1
            return func(x)

        # Use 'trust-constr' for local refinement starting from the best initial sample
        result = minimize(
            wrapped_func,
            best_sample,
            method='trust-constr',
            bounds=bounds,
            options={'maxiter': remaining_budget, 'gtol': 1e-7}  # Enhanced stop condition
        )

        return result.x, result.fun