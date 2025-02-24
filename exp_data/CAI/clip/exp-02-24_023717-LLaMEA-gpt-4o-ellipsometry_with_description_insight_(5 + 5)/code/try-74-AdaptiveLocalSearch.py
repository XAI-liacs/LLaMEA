import numpy as np
from scipy.optimize import minimize

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

        num_initial_samples = min(15, self.budget // 5)
        samples = np.random.uniform(lb, ub, (num_initial_samples, self.dim))

        for sample in samples:
            if evaluations >= self.budget:
                break
            result = self._local_optimize(func, sample, lb, ub)
            evaluations += result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        while evaluations < self.budget:
            # Incorporate simulated annealing for exploration
            result = self._simulated_annealing(func, best_solution, lb, ub)
            evaluations += result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            elif abs(result.fun - best_value) < self.convergence_threshold:
                break

        return best_solution

    def _local_optimize(self, func, start_point, lb, ub):
        step_size = max(0.01, (self.budget - lb[0]) / self.budget)
        return minimize(
            func,
            start_point,
            method='L-BFGS-B',
            bounds=list(zip(lb, ub)),
            options={'maxfun': self.budget, 'eps': step_size}
        )

    # New method for simulated annealing
    def _simulated_annealing(self, func, start_point, lb, ub):
        return minimize(
            func,
            start_point,
            method='L-BFGS-B',
            bounds=list(zip(lb, ub)),
            options={'maxfun': self.budget // 10, 'eps': 0.01}
        )