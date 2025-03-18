import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Ensure initial budget usage for a global search
        initial_samples = int(self.budget * 0.20)  # Adjusted initial sample percentage
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

        # Local optimization using BFGS
        bounds = [(lb[i], ub[i]) for i in range(self.dim)]
        remaining_evaluations = remaining_budget // self.dim
        
        # Use numerical gradient approximation for better refinement
        def wrapped_func(x):
            nonlocal remaining_evaluations
            if remaining_evaluations <= 0:
                raise StopIteration("Budget exceeded")
            remaining_evaluations -= 1
            return func(x)

        # Use multiple local searches from diverse starting points
        results = []
        for _ in range(3):  # Conduct three local searches
            result = minimize(
                wrapped_func,
                best_sample + np.random.uniform(-0.1, 0.1, self.dim),  # Slight perturbation in starting point
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxfun': remaining_budget // 3, 'ftol': 1e-9}  # Split remaining budget
            )
            results.append(result)

        # Choose the best result from multiple local searches
        best_result = min(results, key=lambda res: res.fun)

        return best_result.x, best_result.fun