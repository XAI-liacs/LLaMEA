import numpy as np
from scipy.optimize import minimize

class NaturalComputingOptimizer:
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

        # Local optimization using L-BFGS-B
        if remaining_budget > 0:
            dynamic_maxfun = min(remaining_budget, 100)  # Dynamic adjustment of 'maxfun' based on remaining budget
            result = minimize(func, best_solution, method='L-BFGS-B', bounds=list(zip(lb, ub)), options={'maxfun': dynamic_maxfun, 'ftol': 1e-9})
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        return best_solution