import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        best_solution = None
        best_value = float('inf')

        # Use Sobol sequence for initial sampling
        num_initial_samples = max(1, self.budget // 10)
        sobol = Sobol(d=self.dim, scramble=True)
        initial_samples = sobol.random_base2(m=int(np.log2(num_initial_samples)))
        initial_samples = initial_samples * (func.bounds.ub - func.bounds.lb) + func.bounds.lb

        for sample in initial_samples:
            if self.evaluations >= self.budget:
                break
            solution, value = self.local_search(func, sample, bounds)
            if value < best_value:
                best_solution, best_value = solution, value

        return best_solution

    def local_search(self, func, initial_point, bounds):
        if self.evaluations >= self.budget:
            return initial_point, func(initial_point)

        # Use a local optimizer (BFGS) for fast convergence
        result = minimize(func, initial_point, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - self.evaluations})
        self.evaluations += result.nfev

        return result.x, result.fun