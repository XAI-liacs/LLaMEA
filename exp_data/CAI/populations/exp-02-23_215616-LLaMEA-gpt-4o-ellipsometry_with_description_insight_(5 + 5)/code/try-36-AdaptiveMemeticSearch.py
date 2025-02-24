import numpy as np
from scipy.optimize import minimize, Bounds

class AdaptiveMemeticSearch:
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

        # Set the initial number of global search points
        num_initial_points = max(5, self.budget // 4)
        initial_points = np.random.uniform(
            lower_bounds, upper_bounds, (num_initial_points, self.dim)
        )

        # Adaptive memetic framework
        for point in initial_points:
            if evaluations >= self.budget:
                break

            # Global optimization using a randomized large step search
            global_result = minimize(
                func, point, method='Powell', bounds=bounds,
                options={'maxfev': max(1, (self.budget - evaluations) // 2), 'xtol': 1e-8, 'ftol': 1e-8}
            )
            evaluations += global_result.nfev

            if global_result.success:
                # Dynamic allocation for local search
                remaining_budget = self.budget - evaluations
                local_search_alloc = max(1, remaining_budget // 2)

                # Local optimization using BFGS with dynamic constraints
                local_result = minimize(
                    func, global_result.x, method='BFGS', bounds=bounds,
                    options={'maxiter': local_search_alloc, 'gtol': 1e-8}
                )
                evaluations += local_result.nit

                # Check and update the best solution found
                if local_result.fun < best_value:
                    best_solution = local_result.x
                    best_value = local_result.fun

            else:
                # If global optimization failed, check intermediate result
                if global_result.fun < best_value:
                    best_solution = global_result.x
                    best_value = global_result.fun

        # In case no optimization was successful, return the best initial guess
        if best_solution is None:
            best_solution = initial_points[0]
            best_value = func(best_solution)

        return best_solution, best_value