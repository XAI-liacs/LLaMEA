import numpy as np
from scipy.optimize import minimize

class AdaptiveQuadraticSamplingOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        best_solution = None
        best_value = float('inf')
        remaining_budget = self.budget

        # Initial quadratic sampling
        num_initial_samples = min(self.dim * 3, remaining_budget // 3)
        samples = np.random.uniform(lb, ub, (num_initial_samples, self.dim))
        
        for sample in samples:
            value = func(sample)
            remaining_budget -= 1
            if value < best_value:
                best_value = value
                best_solution = sample
            if remaining_budget <= 0:
                break

        # Adaptive quadratic sampling
        num_adaptive_samples = remaining_budget // 2
        while remaining_budget > num_adaptive_samples:
            center = best_solution
            sigma = 0.1 * (ub - lb)
            adaptive_samples = center + np.random.normal(0, sigma, (num_adaptive_samples, self.dim))
            adaptive_samples = np.clip(adaptive_samples, lb, ub)

            for sample in adaptive_samples:
                value = func(sample)
                remaining_budget -= 1
                if value < best_value:
                    best_value = value
                    best_solution = sample
                if remaining_budget <= 0:
                    break

        # Final refinement using L-BFGS-B
        if remaining_budget > 0:
            result = minimize(func, best_solution, method='L-BFGS-B', bounds=list(zip(lb, ub)), options={'maxfun': remaining_budget, 'ftol': 1e-9})
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        return best_solution