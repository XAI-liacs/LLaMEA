import numpy as np
from scipy.optimize import minimize

class RefinedNaturalComputingOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        best_solution = None
        best_value = float('inf')
        remaining_budget = self.budget

        # Initial uniform sampling
        num_initial_samples = min(self.dim * 5, remaining_budget // 2)
        samples = np.random.uniform(lb, ub, (num_initial_samples, self.dim))
        for sample in samples:
            value = func(sample)
            remaining_budget -= 1
            if value < best_value:
                best_value = value
                best_solution = sample
            if remaining_budget <= 0:
                break

        # Iterative boundary adjustment
        for i in range(self.dim):
            lb[i] = max(lb[i], best_solution[i] - (ub[i] - lb[i]) * 0.1)
            ub[i] = min(ub[i], best_solution[i] + (ub[i] - lb[i]) * 0.1)

        # Local optimization using BFGS
        if remaining_budget > 0:
            result = minimize(func, best_solution, method='BFGS', options={'maxiter': remaining_budget})
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        return best_solution