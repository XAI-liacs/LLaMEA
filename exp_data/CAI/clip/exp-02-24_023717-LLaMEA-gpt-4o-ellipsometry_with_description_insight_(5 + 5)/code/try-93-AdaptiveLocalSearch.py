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

        # Step 1: Uniform sampling for initial guesses with dynamic sample size adjustment
        num_initial_samples = min(10, self.budget // 5)
        sampler = Sobol(d=self.dim, scramble=True)
        samples = lb + (ub - lb) * sampler.random_base2(m=int(np.log2(num_initial_samples)))

        for sample in samples:
            if evaluations >= self.budget:
                break
            gradient = self._estimate_gradient(func, sample)  # Added gradient estimation
            result = self._local_optimize(func, sample + gradient, lb, ub)  # Use gradient for initial guess
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

    def _estimate_gradient(self, func, point, epsilon=1e-8):
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            point_up = np.array(point)
            point_down = np.array(point)
            point_up[i] += epsilon
            point_down[i] -= epsilon
            gradient[i] = (func(point_up) - func(point_down)) / (2 * epsilon)
        return gradient