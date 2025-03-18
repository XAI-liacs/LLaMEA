import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Calculate the number of initial points based on budget
        num_initial_points = min(10, self.budget // 2)
        remaining_budget = self.budget - num_initial_points
        
        # Uniformly sample initial points within bounds
        initial_points = self.uniform_sample_points(func, num_initial_points)
        
        best_solution = None
        best_value = np.inf

        # Evaluate initial points and refine the best ones
        for point in initial_points:
            result = self.optimize_local(func, point, remaining_budget // num_initial_points)
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        return best_solution

    def uniform_sample_points(self, func, num_points):
        lb = func.bounds.lb
        ub = func.bounds.ub
        return [np.random.uniform(lb, ub) for _ in range(num_points)]

    def optimize_local(self, func, initial_point, max_evals):
        bounds = minimize_bounds(func.bounds)
        result = minimize(func, initial_point, method='L-BFGS-B', bounds=bounds, options={'maxfun': max_evals})
        return result

def minimize_bounds(bounds):
    return [(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)]