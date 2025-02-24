import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
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
        num_initial_samples = min(self.budget // 10, 10)
        initial_points = np.random.uniform(lb, ub, (num_initial_samples, self.dim))

        # Evaluate the initial points
        best_point = None
        best_value = float('inf')
        for point in initial_points:
            value = objective(point)
            if value < best_value:
                best_value = value
                best_point = point

        # Optimize using a local method starting from the best initial point
        res = minimize(objective, best_point, method='L-BFGS-B', bounds=list(zip(lb, ub)),
                       options={'maxiter': self.budget - self.evaluations})

        return res.x