import numpy as np
from scipy.optimize import minimize

class ABEL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Initial sampling: Uniform sampling across the parameter space
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        best_solution = lb + (ub - lb) * np.random.rand(self.dim)
        best_value = func(best_solution)
        self.evaluations += 1

        # Adaptive boundary exploration
        while self.evaluations < self.budget * 0.5:  # Allocate half budget for exploration
            candidate = lb + (ub - lb) * np.random.rand(self.dim)
            candidate_value = func(candidate)
            self.evaluations += 1

            if candidate_value < best_value:
                best_solution = candidate
                best_value = candidate_value

        # Local optimization using BFGS
        def local_objective(x):
            nonlocal best_value
            value = func(x)
            self.evaluations += 1
            if value < best_value:
                best_value = value
            return value

        options = {'maxiter': self.budget - self.evaluations, 'disp': False}
        result = minimize(local_objective, best_solution, method='BFGS', bounds=list(zip(lb, ub)), options=options)

        # Return the best solution found
        return result.x, best_value