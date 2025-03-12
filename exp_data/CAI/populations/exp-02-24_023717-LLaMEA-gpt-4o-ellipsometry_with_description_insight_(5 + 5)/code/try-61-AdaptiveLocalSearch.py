import numpy as np
from scipy.optimize import minimize
from skopt import gp_minimize

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

        # Step 1: Bayesian optimization for initial guesses
        def objective(x):
            return func(x)
        
        space = [(lb[i], ub[i]) for i in range(self.dim)]
        result_gp = gp_minimize(objective, space, n_calls=min(15, self.budget // 5)) # Replaced sampling method
        samples = np.array(result_gp.x_iters)

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
        return minimize(
            func,
            start_point,
            method='L-BFGS-B',
            bounds=list(zip(lb, ub)),
            options={'maxfun': self.budget}
        )