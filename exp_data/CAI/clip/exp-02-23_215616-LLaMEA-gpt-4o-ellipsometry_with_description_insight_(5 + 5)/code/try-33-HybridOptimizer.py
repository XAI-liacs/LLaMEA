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

        # Adaptive sampling to balance exploration and exploitation
        num_initial_points = max(5, self.budget // 3)
        initial_points = np.random.uniform(
            lower_bounds, upper_bounds, (num_initial_points, self.dim)
        )

        # Optimize using a hybrid approach
        for point in initial_points:
            if evaluations >= self.budget:
                break

            # Local optimization using adaptive Nelder-Mead
            max_fev_nelder = max(1, (self.budget - evaluations) // 3)
            nelder_mead_result = minimize(
                func, point, method='Nelder-Mead',
                options={'maxfev': max_fev_nelder, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            evaluations += nelder_mead_result.nfev

            if nelder_mead_result.success:
                # Dynamic switch to Powell's method if budget allows
                if evaluations < self.budget:
                    powell_result = minimize(
                        func, nelder_mead_result.x, method='Powell', bounds=bounds,
                        options={'maxfev': self.budget - evaluations, 'xtol': 1e-8}
                    )
                    evaluations += powell_result.nfev

                    # Check and update the best solution found
                    if powell_result.fun < best_value:
                        best_solution = powell_result.x
                        best_value = powell_result.fun
            else:
                # If Nelder-Mead failed, check the intermediate result
                if nelder_mead_result.fun < best_value:
                    best_solution = nelder_mead_result.x
                    best_value = nelder_mead_result.fun

        # In case no optimization was successful, return the best initial guess
        if best_solution is None:
            best_solution = initial_points[0]
            best_value = func(best_solution)

        return best_solution, best_value