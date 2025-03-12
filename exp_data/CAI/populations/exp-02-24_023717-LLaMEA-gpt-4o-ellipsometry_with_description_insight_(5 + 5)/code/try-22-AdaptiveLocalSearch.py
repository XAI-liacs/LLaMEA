import numpy as np
from scipy.optimize import minimize

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

        num_initial_samples = min(10, self.budget // 5)
        samples = np.random.uniform(lb, ub, (num_initial_samples, self.dim))

        for sample in samples:
            if evaluations >= self.budget:
                break
            result = self._local_optimize(func, sample, lb, ub, evaluations)
            evaluations += result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        while evaluations < self.budget:
            result = self._local_optimize(func, best_solution, lb, ub, evaluations)
            evaluations += result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            else:
                break

        return best_solution

    def _local_optimize(self, func, start_point, lb, ub, evaluations):
        radius = (self.budget - evaluations) / self.budget * 0.1  # Shrinking exploration radius
        perturbed_start = np.clip(start_point + np.random.uniform(-radius, radius, self.dim), lb, ub)
        return minimize(
            func,
            perturbed_start,
            method='L-BFGS-B',
            bounds=list(zip(lb, ub)),
            options={'maxfun': self.budget}
        )