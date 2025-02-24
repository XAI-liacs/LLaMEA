import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class AdvancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub

        # Dynamically adjust initial sampling points based on budget and dimensionality
        initial_samples = max(min(self.budget // (2 * self.dim), 100), 10)
        remaining_budget = self.budget - initial_samples

        # Use Sobol sequence for more uniform sampling of initial points
        sobol_sampler = Sobol(d=self.dim, scramble=True)
        samples = sobol_sampler.random_base2(m=int(np.log2(initial_samples)))
        samples = lb + (ub - lb) * samples
        best_value = float('inf')
        best_solution = None

        # Evaluate sampled points
        evaluations = 0
        for sample in samples:
            if evaluations >= self.budget:
                break
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample

        # Define a bounded function to ensure the search remains within the specified bounds
        def bounded_func(x):
            return func(np.clip(x, lb, ub))

        # Use the remaining budget efficiently in local optimization with adaptive L-BFGS-B
        options = {'maxiter': remaining_budget, 'disp': False, 'gtol': 1e-4 * (remaining_budget / self.budget)}
        result = minimize(bounded_func, best_solution, method='L-BFGS-B', bounds=np.array([lb, ub]).T, options=options)

        return result.x