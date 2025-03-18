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
        
        # Initial sampling
        num_initial_points = max(8, self.budget // 8)  # Adjusted number of initial points
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
        
        # Local search and bounds refinement
        while evaluations < self.budget:
            bfgs_optimization(best_solution)
            # Update bounds based on best solution found to refine search
            new_lb = np.maximum(lb, best_solution - 0.12 * (ub - lb))  # Adjusted bound refinement factor
            new_ub = np.minimum(ub, best_solution + 0.12 * (ub - lb))  # Adjusted bound refinement factor
            lb, ub = new_lb, new_ub
            
            # Strategic restart: weighted sampling towards best solution
            restart_point = np.random.uniform(lb, ub, size=self.dim) * 0.9 + best_solution * 0.1
            bfgs_optimization(restart_point)
        
        return best_solution