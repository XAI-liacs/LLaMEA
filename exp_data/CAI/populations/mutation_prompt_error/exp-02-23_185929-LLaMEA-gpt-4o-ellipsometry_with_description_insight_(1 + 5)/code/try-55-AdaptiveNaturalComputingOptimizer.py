import numpy as np
from scipy.optimize import minimize

class AdaptiveNaturalComputingOptimizer:
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
        num_initial_samples = min(self.dim * 5, remaining_budget // 3)
        samples = np.random.uniform(lb, ub, (num_initial_samples, self.dim))
        for sample in samples:
            value = func(sample)
            remaining_budget -= 1
            if value < best_value:
                best_value = value
                best_solution = sample
            if remaining_budget <= 0:
                break

        # Dynamically refine bounds around the best solution found so far
        def refine_bounds(center, scale=0.2):
            new_lb = np.maximum(lb, center - scale * (ub - lb))
            new_ub = np.minimum(ub, center + scale * (ub - lb))
            return new_lb, new_ub

        # Iterative local optimization with adaptive bounds
        while remaining_budget > 0:
            lb_refined, ub_refined = refine_bounds(best_solution)
            result = minimize(func, best_solution, method='L-BFGS-B', bounds=list(zip(lb_refined, ub_refined)), options={'maxfun': remaining_budget, 'ftol': 1e-9})
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            remaining_budget -= result.nfev

        return best_solution