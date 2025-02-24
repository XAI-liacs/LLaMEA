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

        # Enhanced adaptive sampling with a progressive refinement strategy
        num_initial_points = max(5, self.budget // 4)
        initial_points = np.random.uniform(
            lower_bounds, upper_bounds, (num_initial_points, self.dim)
        )

        # Optimize using a hybrid approach with dynamic switching
        for point in initial_points:
            if evaluations >= self.budget:
                break

            # Local optimization using Nelder-Mead
            nm_options = {'maxfev': max(1, (self.budget - evaluations) // 4), 'xatol': 1e-8, 'fatol': 1e-8}
            nelder_mead_result = minimize(func, point, method='Nelder-Mead', options=nm_options)
            evaluations += nelder_mead_result.nfev

            # Dynamic switching: decide whether to proceed with BFGS based on interim results
            current_solution = nelder_mead_result.x if nelder_mead_result.success else point
            current_value = nelder_mead_result.fun if nelder_mead_result.success else func(current_solution)

            # If current evaluation is promising, refine with BFGS if budget allows
            if evaluations < self.budget:
                bfgs_result = minimize(
                    func, current_solution, method='BFGS', bounds=bounds,
                    options={'maxiter': self.budget - evaluations, 'gtol': 1e-8}
                )
                evaluations += bfgs_result.nit

                # Update the best solution found
                if bfgs_result.fun < best_value:
                    best_solution = bfgs_result.x
                    best_value = bfgs_result.fun
            else:
                # If BFGS cannot be executed, check the interim results of Nelder-Mead
                if current_value < best_value:
                    best_solution = current_solution
                    best_value = current_value

        # In case no optimization yielded a better result, use the best initial guess
        if best_solution is None:
            best_solution = initial_points[0]
            best_value = func(best_solution)

        return best_solution, best_value