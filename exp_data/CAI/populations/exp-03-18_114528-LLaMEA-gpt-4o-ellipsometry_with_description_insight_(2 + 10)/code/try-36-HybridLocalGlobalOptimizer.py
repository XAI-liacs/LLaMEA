import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class HybridLocalGlobalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Initial sampling using Sobol sequences
        num_initial_samples = min(15, self.budget // 3)  # Adjusted budget allocation
        sobol_sampler = Sobol(d=self.dim, scramble=False)
        initial_points = sobol_sampler.random_base2(m=int(np.log2(num_initial_samples)))
        initial_points = lb + initial_points * (ub - lb)

        best_solution = None
        best_value = float('inf')
        evaluations = 0

        for point in initial_points:
            # Local optimization using BFGS
            res = minimize(func, point, method='BFGS',
                           options={'maxiter': max(2, (self.budget - evaluations) // num_initial_samples)})  # Refined iteration limit
            evaluations += res.nfev

            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x

            # Check if we have exhausted the budget
            if evaluations >= self.budget:
                break

        return best_solution