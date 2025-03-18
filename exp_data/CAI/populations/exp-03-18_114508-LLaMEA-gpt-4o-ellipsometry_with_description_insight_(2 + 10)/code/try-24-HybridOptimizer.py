import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        best_solution = None
        best_value = float('inf')

        # Improve initial sampling by considering boundary centers
        num_initial_samples = max(1, self.budget // 12)
        initial_samples = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=(num_initial_samples, self.dim))
        initial_samples = np.vstack((initial_samples, (func.bounds.lb + func.bounds.ub) / 2))

        for sample in initial_samples:
            if self.evaluations >= self.budget:
                break
            solution, value = self.local_search(func, sample, bounds)
            if value < best_value:
                best_solution, best_value = solution, value
                # Dynamic bounds adaptation
                bounds = np.array([(max(func.bounds.lb[i], solution[i] - 0.1 * (func.bounds.ub[i] - func.bounds.lb[i])),
                                    min(func.bounds.ub[i], solution[i] + 0.1 * (func.bounds.ub[i] - func.bounds.lb[i])))
                                   for i in range(self.dim)])

        return best_solution

    def local_search(self, func, initial_point, bounds):
        if self.evaluations >= self.budget:
            return initial_point, func(initial_point)

        result = minimize(func, initial_point, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - self.evaluations})
        self.evaluations += result.nfev

        return result.x, result.fun