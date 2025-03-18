import numpy as np
from scipy.optimize import minimize

class HybridLocalGlobalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Adaptive sampling adjustment for initial guesses
        num_initial_samples = min(20, self.budget // 4)  # Adjusted budget allocation
        initial_points = np.random.uniform(lb, ub, (num_initial_samples, self.dim))

        best_solution = None
        best_value = float('inf')
        evaluations = 0

        for point in initial_points:
            # Local optimization using BFGS with dynamic constraint adjustment
            res = minimize(func, point, method='BFGS',
                           bounds=[(lb[i], ub[i]) for i in range(self.dim)],  # Added explicit bounds
                           options={'maxiter': max(3, (self.budget - evaluations) // num_initial_samples),
                                    'gtol': 1e-8})  # Refined iteration limit and step reduction
            evaluations += res.nfev

            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x

            # Check if we have exhausted the budget
            if evaluations >= self.budget:
                break

        return best_solution