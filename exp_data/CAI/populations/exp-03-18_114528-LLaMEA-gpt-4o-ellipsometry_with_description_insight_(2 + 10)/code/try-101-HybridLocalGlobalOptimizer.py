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

        # Initial uniform sampling to get diverse starting points
        num_initial_samples = min(20, self.budget // 3)  # Modified number of initial samples
        initial_points = np.random.uniform(lb, ub, (num_initial_samples, self.dim))

        best_solution = None
        best_value = float('inf')
        evaluations = 0

        for point in initial_points:
            # Local optimization using BFGS
            maxiter = max(2, (self.budget - evaluations) // num_initial_samples)
            res_bfgs = minimize(func, point, method='BFGS',
                                options={'maxiter': maxiter, 'gtol': 1e-5})  # Added gradient tolerance option
            res_nm = minimize(func, point, method='Nelder-Mead', options={'maxiter': maxiter})
            evaluations += res_bfgs.nfev + res_nm.nfev

            # Select the best result from the two methods
            if res_bfgs.fun < res_nm.fun:
                candidate_value, candidate_solution = res_bfgs.fun, res_bfgs.x
            else:
                candidate_value, candidate_solution = res_nm.fun, res_nm.x

            if candidate_value < best_value:
                best_value = candidate_value
                best_solution = candidate_solution

            # Check if we have exhausted the budget
            if evaluations >= self.budget:
                break

        return best_solution