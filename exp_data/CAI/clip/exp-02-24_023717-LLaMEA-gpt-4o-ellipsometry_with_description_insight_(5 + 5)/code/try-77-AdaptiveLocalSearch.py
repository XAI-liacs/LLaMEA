import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol, LatinHypercube

class AdaptiveLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        best_solution = None
        best_value = float('inf')
        evaluations = 0

        # Step 1: Hybrid sampling for initial guesses
        num_initial_samples = min(10, self.budget // 5)
        sobol_sampler = Sobol(d=self.dim, scramble=True)
        lhs_sampler = LatinHypercube(d=self.dim)
        sobol_samples = lb + (ub - lb) * sobol_sampler.random_base2(m=int(np.log2(num_initial_samples // 2)))
        lhs_samples = lb + (ub - lb) * lhs_sampler.random(num_initial_samples // 2)
        samples = np.vstack((sobol_samples, lhs_samples))

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
            else:
                break  # Stop if no improvement

        return best_solution

    def _local_optimize(self, func, start_point, lb, ub):
        return minimize(
            func,
            start_point,
            method='L-BFGS-B',
            bounds=list(zip(lb, ub)),
            options={'maxfun': self.budget}
        )