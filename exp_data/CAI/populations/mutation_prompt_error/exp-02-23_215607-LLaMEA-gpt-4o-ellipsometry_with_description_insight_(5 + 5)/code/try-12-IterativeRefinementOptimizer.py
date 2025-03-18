import numpy as np
from scipy.optimize import minimize

class IterativeRefinementOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Extract bounds from the function
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Define the objective wrapper to count evaluations
        def objective(x):
            if self.evaluations >= self.budget:
                raise Exception("Budget exceeded")
            self.evaluations += 1
            return func(x)
        
        # Uniformly sample initial points
        num_initial_samples = min(self.budget // 20, 20)
        initial_points = np.random.uniform(lb, ub, (num_initial_samples, self.dim))

        # Evaluate initial points to find promising regions
        best_point = None
        best_value = float('inf')
        for point in initial_points:
            value = objective(point)
            if value < best_value:
                best_value = value
                best_point = point
        
        # Iteratively refine the best solution found
        for _ in range(self.budget - self.evaluations):
            # Optimize using a local method (BFGS) starting from the best point found
            res = minimize(objective, best_point, method='L-BFGS-B', bounds=list(zip(lb, ub)))
            if res.success and res.fun < best_value:
                best_value = res.fun
                best_point = res.x
            # Update the search space to focus around the best point
            lb = np.maximum(func.bounds.lb, best_point - (ub - lb) * 0.1)
            ub = np.minimum(func.bounds.ub, best_point + (ub - lb) * 0.1)
        
        return best_point