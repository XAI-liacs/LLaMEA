import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class HybridLocalGlobalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub

        num_initial_samples = min(20, self.budget // 4)
        
        # Use Sobol sequence for better uniformity in initial sampling
        sobol_sampler = qmc.Sobol(d=self.dim, scramble=True)
        initial_points = qmc.scale(sobol_sampler.random(num_initial_samples), lb, ub)

        best_solution = None
        best_value = float('inf')
        evaluations = 0

        for point in initial_points:
            # Local optimization using BFGS
            res = minimize(func, point, method='BFGS',
                           bounds=[(lb[i], ub[i]) for i in range(self.dim)],
                           options={'maxiter': max(3, (self.budget - evaluations) // num_initial_samples),
                                    'gtol': 1e-5})  # Adjusted 'gtol' for better precision
            evaluations += res.nfev

            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x

            if evaluations >= self.budget:
                break

        return best_solution