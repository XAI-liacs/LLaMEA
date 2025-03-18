import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class AdaptiveLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.convergence_threshold = 1e-6

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        best_solution = None
        best_value = float('inf')
        evaluations = 0

        # Step 1: Sobol sequence for initial guesses 
        num_initial_samples = min(15, self.budget // 5)
        sobol_sampler = Sobol(self.dim, scramble=True)
        samples = sobol_sampler.random(num_initial_samples)
        samples = lb + samples * (ub - lb)

        for sample in samples:
            if evaluations >= self.budget:
                break
            result = self._local_optimize(func, sample, lb, ub)
            evaluations += result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        # Step 2: Local optimization starting from the best initial guess
        while evaluations < self.budget:
            result = self._local_optimize(func, best_solution, lb, ub)
            evaluations += result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            elif abs(result.fun - best_value) < self.convergence_threshold:
                break

        return best_solution

    def _local_optimize(self, func, start_point, lb, ub):
        return minimize(
            func,
            start_point,
            method='L-BFGS-B',
            bounds=list(zip(lb, ub)),
            options={'maxfun': self.budget}
        )