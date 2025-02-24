import numpy as np
from scipy.optimize import minimize, Bounds

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lower_bounds = func.bounds.lb
        upper_bounds = func.bounds.ub
        bounds = Bounds(lower_bounds, upper_bounds)
        
        evaluations = 0
        best_solution = None
        best_value = float('inf')

        # Adjust initial sampling density based on budget
        num_initial_points = min(self.budget // 3, 15)
        initial_points = np.random.uniform(
            lower_bounds, upper_bounds, (num_initial_points, self.dim)
        )

        for point in initial_points:
            if evaluations >= self.budget:
                break

            # Reallocate a larger portion of budget to Nelder-Mead
            nelder_mead_result = minimize(
                func, point, method='Nelder-Mead',
                options={'maxfev': max(1, (self.budget - evaluations) // 1.5), 'xatol': 1e-8, 'fatol': 1e-8}
            )
            evaluations += nelder_mead_result.nfev

            if nelder_mead_result.success:
                if evaluations < self.budget:
                    # Dynamic tolerance based on remaining budget
                    dynamic_tol = 1e-8 * (self.budget - evaluations) / self.budget
                    bfgs_result = minimize(
                        func, nelder_mead_result.x, method='BFGS', bounds=bounds,
                        options={'maxiter': self.budget - evaluations, 'gtol': dynamic_tol}
                    )
                    evaluations += bfgs_result.nit

                    if bfgs_result.fun < best_value:
                        best_solution = bfgs_result.x
                        best_value = bfgs_result.fun
            else:
                if nelder_mead_result.fun < best_value:
                    best_solution = nelder_mead_result.x
                    best_value = nelder_mead_result.fun

        if best_solution is None:
            best_solution = initial_points[0]
            best_value = func(best_solution)

        return best_solution, best_value