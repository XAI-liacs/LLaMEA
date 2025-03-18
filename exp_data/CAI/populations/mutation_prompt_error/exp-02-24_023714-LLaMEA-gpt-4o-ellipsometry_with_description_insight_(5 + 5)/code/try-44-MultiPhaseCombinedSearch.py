import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

class MultiPhaseCombinedSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluation_count = 0

    def __call__(self, func):
        initial_samples = 15
        samples = [np.random.uniform(func.bounds.lb, func.bounds.ub) for _ in range(initial_samples)]
        best_solution = None
        best_value = float('inf')

        def wrapped_func(x):
            if self.evaluation_count >= self.budget:
                return float('inf')
            self.evaluation_count += 1
            return func(x)

        # Phase 1: Broad Exploration
        for sample in samples:
            result = minimize(wrapped_func, sample, method='L-BFGS-B', bounds=list(zip(func.bounds.lb, func.bounds.ub)))
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        # Adaptive boundary refinement
        for _ in range(initial_samples // 3):
            norm_samples = norm.rvs(size=self.dim)
            perturbation = 0.1 * (np.array(func.bounds.ub) - np.array(func.bounds.lb)) * norm_samples
            candidate = np.clip(best_solution + perturbation, func.bounds.lb, func.bounds.ub)
            result = minimize(wrapped_func, candidate, method='Nelder-Mead', bounds=list(zip(func.bounds.lb, func.bounds.ub)))

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        # Phase 2: Intensified Exploitation
        remaining_budget = max(0, self.budget - self.evaluation_count)
        if remaining_budget > 0:
            result = minimize(wrapped_func, best_solution, method='L-BFGS-B', bounds=list(zip(func.bounds.lb, func.bounds.ub)),
                              options={'maxiter': remaining_budget})
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        return best_solution