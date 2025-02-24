import numpy as np
from scipy.optimize import minimize
from skopt import Optimizer

class AdaptiveLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.convergence_threshold = 1e-6
        self.optimizer = Optimizer(
            dimensions=[(1.1, 3), (30, 250)],  # Adjusted for fixed problem dimensions
            acq_func="EI",  # Use Expected Improvement for acquisition
            acq_optimizer="sampling"
        )
        
    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        best_solution = None
        best_value = float('inf')
        evaluations = 0

        # Step 1: Bayesian optimization for initial guesses
        num_initial_samples = min(15, self.budget // 5)
        for _ in range(num_initial_samples):
            if evaluations >= self.budget:
                break
            sample = self.optimizer.ask()
            value = func(sample)
            self.optimizer.tell(sample, value)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample

        # Step 2: Local optimization starting from the best initial guess
        while evaluations < self.budget:
            result = self._local_optimize(func, best_solution, lb, ub)
            evaluations += result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            elif abs(result.fun - best_value) < self.convergence_threshold:
                break

        return best_solution

    def _local_optimize(self, func, start_point, lb, ub):
        return minimize(
            func,
            start_point,
            method='L-BFGS-B',
            bounds=list(zip(lb, ub)),
            options={'maxfun': self.budget}
        )