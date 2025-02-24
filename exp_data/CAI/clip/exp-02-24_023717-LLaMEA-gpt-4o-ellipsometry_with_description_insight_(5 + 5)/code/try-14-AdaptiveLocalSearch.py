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

        # Step 1: Uniform sampling for initial guesses
        num_initial_samples = min(10, self.budget // 2)
        samples = np.random.uniform(lb, ub, (num_initial_samples, self.dim))

        # Adjustment factor for dynamic bounds
        bound_shrink_factor = 0.9

        for sample in samples:
            if evaluations >= self.budget:
                break
            result = self._local_optimize(func, sample, lb, ub)
            evaluations += result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        # Step 2: Local optimization with dynamic bounds and refined random perturbation
        while evaluations < self.budget:
            # Apply more precise random perturbations to escape local optima
            perturbed_solution = best_solution + np.random.normal(0, 0.02, self.dim) * (ub - lb)
            perturbed_solution = np.clip(perturbed_solution, lb, ub)

            result = self._local_optimize(func, perturbed_solution, lb, ub)
            evaluations += result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
                # Dynamically adjust the bounds around the new best solution
                lb = np.maximum(lb, best_solution - (best_solution - lb) * bound_shrink_factor)
                ub = np.minimum(ub, best_solution + (ub - best_solution) * bound_shrink_factor)
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