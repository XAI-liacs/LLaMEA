import numpy as np
from scipy.optimize import minimize

class HybridLocalGlobalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        best_solution = None
        best_value = float('inf')

        def local_optimize(x0):
            nonlocal best_solution, best_value
            result = minimize(func, x0, method='L-BFGS-B', bounds=[(lb[i], ub[i]) for i in range(self.dim)])
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        num_local_searches = min(15, max(5, self.budget // 15))  # Adjusted for flexibility
        function_evals_per_search = self.budget // num_local_searches
        
        sampling_density = 0.8 + 0.4 * (self.budget / 100)  # Dynamic sampling density

        for _ in range(num_local_searches):
            x0 = np.random.uniform(lb, ub)
            if np.random.rand() < sampling_density:  # Conditional local search initiation
                local_optimize(x0)

            if self.budget < 0:
                break

        return best_solution