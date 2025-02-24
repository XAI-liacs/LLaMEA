import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def sobol_sampling(self, bounds, num_samples):
        sampler = Sobol(d=self.dim, scramble=True)
        sample_points = sampler.random_base2(m=int(np.log2(num_samples)))
        samples = []
        for i, (lb, ub) in enumerate(zip(bounds.lb, bounds.ub)):
            samples.append(lb + (ub - lb) * sample_points[:, i])
        return np.array(samples).T

    def local_optimization(self, func, x0, bounds):
        result = minimize(func, x0, method='Nelder-Mead')  # Changed method
        return result.x, result.fun

    def adjust_bounds(self, best_solution, bounds, reduction_factor=0.9):
        new_bounds = []
        for i, (lb, ub) in enumerate(bounds):
            center = best_solution[i]
            range_half = (ub - lb) * reduction_factor / 2
            new_lb = max(lb, center - range_half)
            new_ub = min(ub, center + range_half)
            new_bounds.append((new_lb, new_ub))
        return new_bounds

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

        if best_solution is not None:
            bounds = self.adjust_bounds(best_solution, bounds)

            for x0 in initial_points:
                if self.budget <= 0:
                    break

                x, value = self.local_optimization(func, x0, bounds)
                self.budget -= 1

                if value < best_value:
                    best_value = value
                    best_solution = x

        return best_solution