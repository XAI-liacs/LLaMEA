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
        return qmc.scale(sample, lb * (1 - (self.evals + 5) / self.budget), ub * (1 + (self.evals + 5) / self.budget))

    def refine_solution(self, x0, func, bounds):
        options = {'maxiter': self.budget - self.evals, 'learning_rate': 0.5}  # Changed line
        result = minimize(func, x0, method='L-BFGS-B', bounds=bounds, options=options)  
        self.evals += result.nfev
        return result.x, result.fun

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        num_initial_samples = min(max(10, self.budget // 15), int(self.budget * 0.1))
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

            diversity_factor = np.std(initial_solutions)
            penalty_factor = 0.15 * (1 - self.evals / self.budget) * (best_value / (best_value + 1) + 2 * diversity_factor)
            lb = np.maximum(lb, refined_solution - (0.1 + penalty_factor) * (ub - lb))
            ub = np.minimum(ub, refined_solution + (0.1 + penalty_factor) * (ub - lb))

        return best_solution