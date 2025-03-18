import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def uniform_sampling(self, lb, ub, num_samples):
        sampler = qmc.LatinHypercube(d=len(lb))
        sample = sampler.random(n=num_samples)
        return qmc.scale(sample, lb, ub)

    def refine_solution(self, x0, func, bounds):
        options = {'maxiter': self.budget - self.evals}
        result = minimize(func, x0, method='L-BFGS-B', options=options, bounds=bounds)  # Changed line
        self.evals += result.nfev
        return result.x, result.fun

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        num_initial_samples = min(25, self.budget // 10)  # Changed line
        initial_solutions = self.uniform_sampling(lb, ub, num_initial_samples)
        best_solution = None
        best_value = float('inf')

        for x0 in initial_solutions:
            if self.evals >= self.budget:
                break
            refined_solution, refined_value = self.refine_solution(x0, func, list(zip(lb, ub)))
            if refined_value < best_value:
                best_value = refined_value
                best_solution = refined_solution

            penalty_factor = 0.1 * (self.budget / (self.evals + 1))
            lb = np.maximum(lb, refined_solution - (0.1 + penalty_factor) * (ub - lb))
            ub = np.minimum(ub, refined_solution + (0.1 + penalty_factor) * (ub - lb))

        return best_solution