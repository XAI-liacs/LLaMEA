import numpy as np
from scipy.optimize import minimize

class HybridLocalGlobalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub

        num_initial_samples = min(20, self.budget // 4)
        initial_points = np.random.uniform(lb, ub, (num_initial_samples, self.dim))

        best_solution = None
        best_value = float('inf')
        evaluations = 0

        for point in initial_points:
            # Introducing dynamic mutation before local optimization
            mutated_point = np.clip(point + np.random.normal(0, 0.1, self.dim), lb, ub)

            res = minimize(func, mutated_point, method='BFGS',
                           bounds=[(lb[i], ub[i]) for i in range(self.dim)],
                           options={'maxiter': max(3, (self.budget - evaluations) // num_initial_samples)})
            evaluations += res.nfev

            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x

            if evaluations >= self.budget:
                break

        return best_solution