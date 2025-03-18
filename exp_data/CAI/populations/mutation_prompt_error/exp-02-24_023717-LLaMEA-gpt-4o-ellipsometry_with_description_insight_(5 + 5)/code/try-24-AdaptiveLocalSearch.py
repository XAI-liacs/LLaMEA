import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

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

        # Step 1: Sobol sequence for initial guesses with dynamic sample size adjustment
        num_initial_samples = min(10, self.budget // 5)
        sobol = Sobol(d=self.dim)
        samples = sobol.random_base2(m=int(np.log2(num_initial_samples))) * (ub - lb) + lb

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
            dynamic_bounds = [
                (max(l, b - 0.1 * (u - l)), min(u, b + 0.1 * (u - l)))
                for b, l, u in zip(best_solution, lb, ub)
            ]
            result = self._local_optimize(func, best_solution, *zip(*dynamic_bounds))
            evaluations += result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            elif evaluations + self.dim >= self.budget:
                break  # Stop if no improvement or budget is nearly exhausted

        return best_solution

    def _local_optimize(self, func, start_point, lb, ub):
        return minimize(
            func,
            start_point,
            method='L-BFGS-B',
            bounds=list(zip(lb, ub)),
            options={'maxfun': self.budget}
        )