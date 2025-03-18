import numpy as np
from scipy.optimize import minimize

class AdaptiveLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')
        
        # Initial sampling with adaptive sampling strategy
        num_initial_points = max(7, self.budget // 8)
        initial_samples = np.random.uniform(lb, ub, (num_initial_points, self.dim))
        
        evaluations = 0
        
        # Evaluate initial points
        for sample in initial_samples:
            if evaluations >= self.budget:
                break
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample
        
        # BFGS Optimization from best initial point
        def bfgs_optimization(start_point):
            nonlocal evaluations, best_value, best_solution
            bounds = [(low, high) for low, high in zip(lb, ub)]
            result = minimize(func, start_point, method='L-BFGS-B', bounds=bounds)
            evaluations += result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
        
        # Dual-strategy local search and bounds refinement
        while evaluations < self.budget:
            bfgs_optimization(best_solution)
            # New strategy to refine search using dual approach
            if evaluations < self.budget // 2:
                new_lb = np.maximum(lb, best_solution - 0.05 * (ub - lb))
                new_ub = np.minimum(ub, best_solution + 0.05 * (ub - lb))
            else:
                new_lb = np.maximum(lb, best_solution - 0.15 * (ub - lb))
                new_ub = np.minimum(ub, best_solution + 0.15 * (ub - lb))

            lb, ub = new_lb, new_ub
            
            # Random restart or strategic perturbation within updated bounds
            if np.random.rand() < 0.5:
                restart_point = np.random.uniform(lb, ub)
            else:
                restart_point = best_solution + np.random.normal(0, 0.01, self.dim)  # Perturbation strategy
            bfgs_optimization(restart_point)
        
        return best_solution