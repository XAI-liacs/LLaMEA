import numpy as np
from scipy.optimize import minimize, Bounds

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds from the function
        lower_bounds = func.bounds.lb
        upper_bounds = func.bounds.ub
        bounds = Bounds(lower_bounds, upper_bounds)

        # Initialize variables
        evaluations = 0
        best_solution = None
        best_value = float('inf')

        # Uniform sampling to get initial points
        num_initial_points = min(self.budget // 2, 15)
        initial_points = np.random.uniform(
            lower_bounds, upper_bounds, (num_initial_points, self.dim)
        )

        # Optimize using a hybrid approach
        success_rate = 0  # Track success rate
        for point in initial_points:
            if evaluations >= self.budget:
                break

            # Choose local optimization method based on success rate
            method = 'Nelder-Mead' if success_rate < 0.5 else 'Powell'
            local_result = minimize(
                func, point, method=method,
                options={'maxfev': max(1, (self.budget - evaluations) // 3), 'xatol': 1e-8, 'fatol': 1e-8}
            )
            evaluations += local_result.nfev
            success_rate = local_result.success

            if local_result.success:
                # Further refine with BFGS if budget allows
                if evaluations < self.budget:
                    bfgs_result = minimize(
                        func, local_result.x, method='BFGS', bounds=bounds,
                        options={'maxiter': self.budget - evaluations, 'gtol': 1e-8}
                    )
                    evaluations += bfgs_result.nit

                    # Check and update the best solution found
                    if bfgs_result.fun < best_value:
                        best_solution = bfgs_result.x
                        best_value = bfgs_result.fun
            else:
                # If local optimization failed, check the intermediate result
                if local_result.fun < best_value:
                    best_solution = local_result.x
                    best_value = local_result.fun

        # In case no optimization was successful, return the best initial guess
        if best_solution is None:
            best_solution = initial_points[0]
            best_value = func(best_solution)

        return best_solution, best_value