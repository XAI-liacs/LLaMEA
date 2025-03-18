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
        num_initial_points = max(5, self.budget // 10)  # Changed sampling ratio for better initial diversity
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
        adaptive_restart_prob = 0.05  # Introduced adaptive restart probability
        while evaluations < self.budget:
            bfgs_optimization(best_solution)
            # Update bounds based on best solution found to refine search
            new_lb = np.maximum(lb, best_solution - 0.15 * (ub - lb))  # Improved refinement width
            new_ub = np.minimum(ub, best_solution + 0.15 * (ub - lb))
            lb, ub = new_lb, new_ub
            
            # Random restart within updated bounds
            restart_point = np.random.uniform(lb, ub)
            bfgs_optimization(restart_point)
            
            # Introduce dynamic restart strategy
            if np.random.rand() < adaptive_restart_prob:  # Adaptive restart frequency
                restart_point = np.random.uniform(func.bounds.lb, func.bounds.ub)
                bfgs_optimization(restart_point)
                adaptive_restart_prob = min(0.2, adaptive_restart_prob + 0.01)  # Increase probability
            else:
                adaptive_restart_prob = max(0.02, adaptive_restart_prob - 0.005)  # Decrease probability
        
        return best_solution