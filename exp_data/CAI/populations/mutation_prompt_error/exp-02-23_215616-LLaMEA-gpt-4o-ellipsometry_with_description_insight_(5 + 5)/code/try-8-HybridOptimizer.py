import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.stats.qmc import Sobol

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

        # Sobol sequence for deterministic initial points
        num_initial_points = min(self.budget // 2, 10)
        sobol_sampler = Sobol(d=self.dim, scramble=False)
        initial_points = sobol_sampler.random_base2(m=int(np.log2(num_initial_points))) * (upper_bounds - lower_bounds) + lower_bounds

        # Optimize using a hybrid approach
        for point in initial_points:
            if evaluations >= self.budget:
                break

            # Local optimization using Nelder-Mead
            nelder_mead_result = minimize(
                func, point, method='Nelder-Mead',
                options={'maxfev': max(1, (self.budget - evaluations) // 4), 'xatol': 1e-8, 'fatol': 1e-8}
            )
            evaluations += nelder_mead_result.nfev

            if nelder_mead_result.success:
                # Further refine with BFGS if budget allows
                if evaluations < self.budget:
                    bfgs_result = minimize(
                        func, nelder_mead_result.x, method='BFGS', bounds=bounds,
                        options={'maxiter': self.budget - evaluations, 'gtol': 1e-8}
                    )
                    evaluations += bfgs_result.nit

                    # Check and update the best solution found
                    if bfgs_result.fun < best_value:
                        best_solution = bfgs_result.x
                        best_value = bfgs_result.fun
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