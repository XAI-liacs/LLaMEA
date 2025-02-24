import numpy as np
from scipy.optimize import minimize

class DASEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Define search space dimensionality
        space_dim = len(lb)

        # Stage 1: Coarse Sampling
        coarse_samples = int(self.budget * 0.3)  # Use 30% of budget for coarse sampling
        sampled_points = np.random.uniform(lb, ub, size=(coarse_samples, space_dim))

        # Evaluate sampled points and track the best
        best_val = float('inf')
        best_point = None
        evaluations = 0

        for point in sampled_points:
            value = func(point)
            evaluations += 1
            if value < best_val:
                best_val = value
                best_point = point

        # Stage 2: Refined Exploitation with Sequential Quadratic Programming (SQP)
        remaining_budget = self.budget - evaluations

        def limited_func(x):
            nonlocal evaluations
            if evaluations < self.budget:
                evaluations += 1
                return func(x)
            else:
                raise Exception("Exceeded budget")

        # Optimize using SQP starting from the best coarse sample
        constraints = [{'type': 'ineq', 'fun': lambda x: x - lb},
                       {'type': 'ineq', 'fun': lambda x: ub - x}]
        
        res = minimize(limited_func, best_point, method='SLSQP', bounds=list(zip(lb, ub)), constraints=constraints)

        # Return the best found solution within budget
        return res.x, res.fun