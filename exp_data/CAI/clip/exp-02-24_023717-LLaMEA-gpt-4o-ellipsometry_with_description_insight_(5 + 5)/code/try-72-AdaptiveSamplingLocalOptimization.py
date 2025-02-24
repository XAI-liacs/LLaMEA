import numpy as np
from scipy.optimize import minimize

class AdaptiveSamplingLocalOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        best_solution = None
        best_value = float('inf')
        evaluations = 0

        # Step 1: Initial adaptive sampling
        initial_sample_size = min(10, self.budget // 5)
        samples = np.random.uniform(lb, ub, (initial_sample_size, self.dim))

        for sample in samples:
            if evaluations >= self.budget:
                break
            result = self._local_optimize(func, sample, lb, ub)
            evaluations += result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        # Step 2: Iterative local optimization with stagnation detection
        stagnation_threshold = 1e-6
        stagnation_count = 0

        while evaluations < self.budget:
            result = self._local_optimize(func, best_solution, lb, ub)
            evaluations += result.nfev

            if result.fun < best_value:
                if abs(best_value - result.fun) < stagnation_threshold:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                best_value = result.fun
                best_solution = result.x

            if stagnation_count >= 3:
                break  # Stop if stagnation detected

        return best_solution

    def _local_optimize(self, func, start_point, lb, ub):
        return minimize(
            func,
            start_point,
            method='L-BFGS-B',
            bounds=list(zip(lb, ub)),
            options={'maxfun': min(self.budget, 100)}
        )