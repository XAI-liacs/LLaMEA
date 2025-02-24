import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class AdvancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub

        # Use Sobol sequence for better initial exploration
        initial_samples = max(min(self.budget // (3 * self.dim), 100), 10)
        remaining_budget = self.budget - initial_samples

        # Generate Sobol samples within the bounds
        sobol_sampler = Sobol(d=self.dim, scramble=True)
        samples = sobol_sampler.random_base2(int(np.log2(initial_samples)))
        samples = lb + (ub - lb) * samples
        best_value = float('inf')
        best_solution = None

        evaluations = 0
        for sample in samples:
            if evaluations >= self.budget:
                break
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample

        def bounded_func(x):
            return func(np.clip(x, lb, ub))

        # Adjust local optimizer settings based on convergence feedback
        adjusted_budget = max(10, int(remaining_budget * (1.0 - best_value)))  # Adjust budget
        options = {'maxiter': adjusted_budget, 'disp': False}
        result = minimize(bounded_func, best_solution, method='L-BFGS-B', bounds=np.array([lb, ub]).T, options=options)

        return result.x