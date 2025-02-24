import numpy as np
from scipy.optimize import minimize

class AdaptiveLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.convergence_threshold = 1e-6 # New adaptive convergence threshold

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        best_solution = None
        best_value = float('inf')
        evaluations = 0

        # Step 1: Uniform sampling for initial guesses with refined dynamic sample size
        num_initial_samples = min(15, self.budget // 5) # Increased initial sample size
        samples = np.random.uniform(lb, ub, (num_initial_samples, self.dim))

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
            elif abs(result.fun - best_value) < self.convergence_threshold: # Added check for adaptive convergence
                break  # Stop if no significant improvement

        return best_solution

    def _local_optimize(self, func, start_point, lb, ub):
        progress_factor = min(1, max(0.01, (self.budget - lb[0]) / self.budget))  # Adaptive step size with progress factor
        step_size = progress_factor * 0.1  # Modify step size based on progress factor
        return minimize(
            func,
            start_point,
            method='L-BFGS-B',
            bounds=list(zip(lb, ub)),
            options={'maxfun': self.budget, 'eps': step_size} # Applied adaptive step size
        )