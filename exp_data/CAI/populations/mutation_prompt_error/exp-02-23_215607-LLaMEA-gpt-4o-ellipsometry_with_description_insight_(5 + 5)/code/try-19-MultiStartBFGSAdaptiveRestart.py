import numpy as np
from scipy.optimize import minimize

class MultiStartBFGSAdaptiveRestart:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_restarts = max(1, int(np.log10(budget)))  # Adaptive number of restarts

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        evals = 0
        best_sample = None
        best_value = float('inf')

        def wrapped_func(x):
            nonlocal evals
            if evals >= self.budget:
                return float('inf')
            value = func(x)
            evals += 1
            return value

        for _ in range(self.num_restarts):
            if evals >= self.budget:
                break

            # Initial random start within bounds
            initial_sample = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)

            # Local optimization using BFGS
            result = minimize(wrapped_func, initial_sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - evals})
            if result.fun < best_value:
                best_value = result.fun
                best_sample = result.x

            # Adaptive restart decision
            if evals / self.budget > 0.6 and best_value < 1e-4:  # Conservative convergence threshold
                break

        return best_sample