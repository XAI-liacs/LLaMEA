import numpy as np
from scipy.optimize import minimize

class ProgressiveAnnealingSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        best_solution = None
        best_value = float('inf')
        evaluations = 0

        # Initial uniform sampling for starting points
        num_initial_samples = min(10, self.budget // 3)
        samples = np.random.uniform(lb, ub, (num_initial_samples, self.dim))

        # Cooling schedule parameters
        temp_initial = 1.0
        temp_final = 0.01
        cooling_rate = 0.95

        current_temp = temp_initial

        for sample in samples:
            if evaluations >= self.budget:
                break
            result = self._local_optimize(func, sample, lb, ub)
            evaluations += result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        # Progressive refinement and annealed perturbation
        while evaluations < self.budget:
            # Annealed random perturbation
            perturbation_scale = current_temp * (ub - lb)
            perturbed_solution = best_solution + np.random.normal(0, perturbation_scale, self.dim)
            perturbed_solution = np.clip(perturbed_solution, lb, ub)

            result = self._local_optimize(func, perturbed_solution, lb, ub)
            evaluations += result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
                # Refine bounds using narrower search space around the best solution
                lb = np.maximum(lb, best_solution - perturbation_scale / 2)
                ub = np.minimum(ub, best_solution + perturbation_scale / 2)
            else:
                if evaluations < self.budget * 0.75:
                    continue
                break  # Stop if no improvement

            # Update temperature
            current_temp = max(temp_final, current_temp * cooling_rate)

        return best_solution

    def _local_optimize(self, func, start_point, lb, ub):
        return minimize(
            func,
            start_point,
            method='L-BFGS-B',
            bounds=list(zip(lb, ub)),
            options={'maxfun': self.budget}
        )