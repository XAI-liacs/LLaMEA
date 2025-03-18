import numpy as np
from scipy.optimize import minimize

class HybridLocalGlobalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize the bounds and dimensionality
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Determine the number of initial samples based on budget
        num_samples = min(10, self.budget // 2)  # Use half budget for initial exploration

        # Uniformly sample initial points
        initial_samples = np.random.uniform(low=lb, high=ub, size=(num_samples, self.dim))
        best_solution = None
        best_value = np.inf

        evaluations = 0
        
        # Evaluate initial samples
        for sample in initial_samples:
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample

        # Use remaining evaluations for local optimization from best initial sample
        remaining_budget = self.budget - evaluations

        if remaining_budget > 0:
            result = minimize(
                func,
                best_solution,
                bounds=[(lb[i], ub[i]) for i in range(self.dim)],
                method='L-BFGS-B',
                options={'maxfun': remaining_budget, 'ftol': 1e-9, 'eps': 1e-6}  # Add early stopping criteria
            )

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        return best_solution