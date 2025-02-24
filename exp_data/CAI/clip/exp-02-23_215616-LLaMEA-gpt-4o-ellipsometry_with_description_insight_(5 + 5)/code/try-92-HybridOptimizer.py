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

        # Adjusted enhanced uniform sampling to get initial points
        num_initial_points = min(self.budget // (self.dim * 2), 15)
        initial_points = np.random.uniform(
            lower_bounds, upper_bounds, (num_initial_points, self.dim)
        )

        # Optimize using a hybrid approach
        for point in initial_points:
            if evaluations >= self.budget:
                break

            # Local optimization using BFGS if budget allows, for faster convergence
            if evaluations < self.budget:
                bfgs_result = minimize(
                    func, point, method='BFGS', bounds=bounds,
                    options={'maxiter': self.budget - evaluations, 'gtol': 1e-8}
                )
                evaluations += bfgs_result.nit

                # Check and update the best solution found
                if bfgs_result.fun < best_value:
                    best_solution = bfgs_result.x
                    best_value = bfgs_result.fun

            # If BFGS budget is exceeded, fallback to Nelder-Mead
            if evaluations < self.budget:
                nelder_mead_result = minimize(
                    func, bfgs_result.x, method='Nelder-Mead',
                    options={'maxfev': max(1, (self.budget - evaluations) // 3), 'xatol': 1e-9, 'fatol': 1e-9}
                )
                evaluations += nelder_mead_result.nfev

                if nelder_mead_result.success:
                    if nelder_mead_result.fun < best_value:
                        best_solution = nelder_mead_result.x
                        best_value = nelder_mead_result.fun
                else:
                    if nelder_mead_result.fun < best_value:
                        best_solution = nelder_mead_result.x
                        best_value = nelder_mead_result.fun

        # In case no optimization was successful, return the best initial guess
        if best_solution is None:
            best_solution = initial_points[0]
            best_value = func(best_solution)

        return best_solution, best_value