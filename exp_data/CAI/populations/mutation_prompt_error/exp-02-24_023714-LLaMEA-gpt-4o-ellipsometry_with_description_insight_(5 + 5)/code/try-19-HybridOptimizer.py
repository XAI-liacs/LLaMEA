import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def sobol_sampling(self, bounds, num_samples):
        sampler = Sobol(d=self.dim, scramble=True)
        samples = sampler.random_base2(m=int(np.log2(num_samples)))
        scaled_samples = np.array([lb + (ub - lb) * samples[:, i] for i, (lb, ub) in enumerate(zip(bounds.lb, bounds.ub))]).T
        return scaled_samples

    def local_optimization(self, func, x0, bounds):
        result = minimize(func, x0, method='L-BFGS-B', bounds=bounds)
        return result.x, result.fun

    def __call__(self, func):
        num_initial_samples = min(10, self.budget // 2)
        initial_points = self.sobol_sampling(func.bounds, num_initial_samples)
        
        best_solution = None
        best_value = float('inf')

        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        
        for x0 in initial_points:
            if self.budget <= 0:
                break

            x, value = self.local_optimization(func, x0, bounds)
            self.budget -= 1  # Counting the local optimization as a single budget usage

            if value < best_value:
                best_value = value
                best_solution = x

        return best_solution