import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class HybridLocalGlobalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Sobol sequence sampling for initial points
        sobol_sampler = qmc.Sobol(d=self.dim)
        initial_points = qmc.scale(sobol_sampler.random_base2(m=int(np.log2(min(10, self.budget // 4)))), lb, ub)

        best_solution = None
        best_value = float('inf')
        evaluations = 0

        for point in initial_points:
            # Dynamic choice of optimization method
            method = 'BFGS' if self.dim == 2 else 'Nelder-Mead'
            # Local optimization using selected method
            res = minimize(func, point, method=method,
                           bounds=[(lb[i], ub[i]) for i in range(self.dim)],
                           options={'maxiter': max(1, (self.budget - evaluations) // len(initial_points))})
            evaluations += res.nfev

            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x

            # Check if we have exhausted the budget
            if evaluations >= self.budget:
                break

        return best_solution