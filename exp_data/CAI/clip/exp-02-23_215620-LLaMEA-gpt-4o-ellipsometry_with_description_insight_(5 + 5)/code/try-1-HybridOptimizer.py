import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Define search space dimensionality
        space_dim = len(lb)
        
        # Calculate number of initial samples
        num_samples = int(self.budget * 0.2)  # Use 20% of budget for initial sampling

        # Uniformly sample initial points in the parameter space
        initial_points = np.random.uniform(lb, ub, size=(num_samples, space_dim))
        
        # Evaluate sampled points and keep track of the best one
        best_val = float('inf')
        best_point = None
        evaluations = 0

        for point in initial_points:
            value = func(point)
            evaluations += 1
            if value < best_val:
                best_val = value
                best_point = point
        
        # Remaining budget for local search
        remaining_budget = self.budget - evaluations

        # Define BFGS optimization with constraint to not exceed evaluations
        def limited_func(x):
            nonlocal evaluations
            if evaluations < self.budget:
                evaluations += 1
                return func(x)
            else:
                raise Exception("Exceeded budget")

        # Optimize using BFGS from the best initial point
        res = minimize(limited_func, best_point, method='L-BFGS-B', bounds=list(zip(lb, ub)))

        # Return the best found solution within budget
        return res.x, res.fun