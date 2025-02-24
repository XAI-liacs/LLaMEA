import numpy as np
from scipy.optimize import minimize

class ALSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0
    
    def __call__(self, func):
        # Step 1: Initialize uniform sampling for starting points
        lb, ub = func.bounds.lb, func.bounds.ub
        num_initial_points = min(20, self.budget // 3)  # Changed line
        initial_points = np.random.uniform(lb, ub, (num_initial_points, self.dim))
        
        # Step 2: Evaluate initial points and select the best
        best_point, best_value = None, float('inf')
        for point in initial_points:
            if self.eval_count >= self.budget:
                break
            value = func(point)
            self.eval_count += 1
            if value < best_value:
                best_value, best_point = value, point

        # Step 3: Local optimization using BFGS from the best initial point
        def limited_func(x):
            nonlocal best_value
            if self.eval_count >= self.budget:
                return best_value
            value = func(x)
            self.eval_count += 1
            if value < best_value:
                best_value = value
            return value

        # Use adaptive bounds based on best_point
        adaptive_bounds = [(max(lb[i], best_point[i] - 0.15*(ub[i]-lb[i])),  # Changed line
                            min(ub[i], best_point[i] + 0.15*(ub[i]-lb[i]))) for i in range(self.dim)]  # Changed line
        
        result = minimize(limited_func, best_point, method='L-BFGS-B', bounds=adaptive_bounds, options={'maxfun': self.budget - self.eval_count})
        
        return result.x, result.fun