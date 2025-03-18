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
        num_initial_samples = min(25, self.budget // 4)  # Adjusted number of initial samples
        initial_points = np.random.uniform(lb, ub, (num_initial_samples, self.dim))

        best_solution = None
        best_value = float('inf')
        evaluations = 0

        for point in initial_points:
            # Local optimization using BFGS with dynamic bounds adjustment
            current_lb = lb + (point - lb) * 0.1  # Added dynamic refinement
            current_ub = ub - (ub - point) * 0.1  # Added dynamic refinement
            clipped_point = np.clip(point, current_lb, current_ub)
            
            maxiter = max(3, (self.budget - evaluations) // num_initial_samples)
            res = minimize(func, clipped_point, method='BFGS',
                           bounds=np.array([current_lb, current_ub]).T,  # Applied bounds
                           options={'maxiter': maxiter, 'gtol': 1e-5})
            evaluations += res.nfev

            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x

            if evaluations >= self.budget:
                break

        return best_solution