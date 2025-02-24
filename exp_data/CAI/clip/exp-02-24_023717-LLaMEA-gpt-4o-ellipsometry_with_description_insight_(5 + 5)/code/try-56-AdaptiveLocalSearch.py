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
        stagnation_counter = 0  # Added for early stopping

        # Step 1: Uniform sampling for initial guesses with dynamic sample size adjustment
        num_initial_samples = min(10, self.budget // 5)
        samples = np.random.uniform(lb, ub, (num_initial_samples, self.dim))

        for sample in samples:
            if evaluations >= self.budget:
                break
            result = self._local_optimize(func, sample, lb, ub)
            evaluations += result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
                stagnation_counter = 0  # Reset stagnation counter on improvement
            else:
                stagnation_counter += 1  # Increment on no improvement

        # Step 2: Local optimization starting from the best initial guess
        while evaluations < self.budget:
            if stagnation_counter > 2:  # Early stopping if stagnation occurs
                break
            result = self._local_optimize(func, best_solution, lb, ub)
            evaluations += result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
                stagnation_counter = 0  # Reset on improvement
            else:
                stagnation_counter += 1  # Increment on no improvement

        return best_solution

    def _local_optimize(self, func, start_point, lb, ub):
        return minimize(
            func,
            start_point,
            method='L-BFGS-B',
            bounds=list(zip(lb, ub)),
            options={'maxfun': self.budget}
        )