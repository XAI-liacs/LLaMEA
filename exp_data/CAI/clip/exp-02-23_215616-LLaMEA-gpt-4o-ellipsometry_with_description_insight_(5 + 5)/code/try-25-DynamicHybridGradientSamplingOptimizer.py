import numpy as np
from scipy.optimize import minimize, Bounds

class DynamicHybridGradientSamplingOptimizer:
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
        num_initial_points = min(self.budget // 3, 10)
        initial_points = np.random.uniform(
            lower_bounds, upper_bounds, (num_initial_points, self.dim)
        )

        # Gradient sampling-based local search function
        def gradient_sampling_search(x, remaining_budget):
            if remaining_budget <= 0:
                return x, func(x)

            # Small perturbations around the current point
            perturbations = np.random.normal(0, 0.01, (5, self.dim))
            neighbors = x + perturbations
            neighbors = np.clip(neighbors, lower_bounds, upper_bounds)

            best_local_solution = x
            best_local_value = func(x)

            for neighbor in neighbors:
                if evaluations >= self.budget:
                    break
                neighbor_value = func(neighbor)
                evaluations += 1

                if neighbor_value < best_local_value:
                    best_local_solution = neighbor
                    best_local_value = neighbor_value

            return best_local_solution, best_local_value

        # Optimize using the hybrid approach
        for point in initial_points:
            if evaluations >= self.budget:
                break

            # Local optimization using gradient sampling
            point, value = gradient_sampling_search(point, (self.budget - evaluations) // 2)

            # Further optimize using Nelder-Mead and BFGS if budget allows
            if evaluations < self.budget:
                nelder_mead_result = minimize(
                    func, point, method='Nelder-Mead',
                    options={'maxfev': (self.budget - evaluations) // 3, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                evaluations += nelder_mead_result.nfev
                
                if nelder_mead_result.success:
                    point = nelder_mead_result.x
                    value = nelder_mead_result.fun

                if evaluations < self.budget:
                    bfgs_result = minimize(
                        func, point, method='BFGS', bounds=bounds,
                        options={'maxiter': self.budget - evaluations, 'gtol': 1e-8}
                    )
                    evaluations += bfgs_result.nit

                    if bfgs_result.fun < value:
                        point = bfgs_result.x
                        value = bfgs_result.fun

            # Update the best solution found
            if value < best_value:
                best_solution = point
                best_value = value

        # In case no optimization was successful, return the best initial guess
        if best_solution is None:
            best_solution = initial_points[0]
            best_value = func(best_solution)

        return best_solution, best_value