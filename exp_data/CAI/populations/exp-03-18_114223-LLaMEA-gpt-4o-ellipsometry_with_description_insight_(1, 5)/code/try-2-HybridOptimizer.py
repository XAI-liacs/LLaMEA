import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def uniform_sampling(self, lb, ub, num_samples):
        return np.random.uniform(lb, ub, (num_samples, len(lb)))

    def refine_solution(self, x0, func, bounds):
        # Adjust step size based on remaining budget
        options = {'maxiter': self.budget - self.evals, 'xatol': 1e-4 * (self.budget - self.evals) / self.budget}
        result = minimize(func, x0, method='Nelder-Mead', options=options, bounds=bounds)
        self.evals += result.nfev
        return result.x, result.fun

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        num_initial_samples = min(10, self.budget // 10)
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

            # Adaptive boundary rescaling
            lb = np.maximum(lb, refined_solution - 0.1 * (ub - lb))
            ub = np.minimum(ub, refined_solution + 0.1 * (ub - lb))

        return best_solution