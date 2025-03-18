import numpy as np
from scipy.optimize import minimize
from skopt import gp_minimize
from pyDOE2 import lhs

class AdaptiveSamplingLocalRefinement:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.grid_samples = min(10, budget // 3)  # Adjusted initial sample size

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        best_solution = None
        best_value = float('inf')
        
        # Orthogonal Latin Hypercube Sampling
        grid_points = lb + (ub - lb) * lhs(self.dim, samples=self.grid_samples)  # Diverse initial samples
        for i in range(self.grid_samples):
            x0 = grid_points[i]
            value = func(x0)
            self.evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = x0
                # Hybrid local refinement using Bayesian Optimization
                if self.evaluations < self.budget // 2:
                    res = gp_minimize(func, [(lb[i], ub[i]) for i in range(self.dim)], n_calls=self.budget // 15, x0=[x0])
                    if res.fun < best_value:
                        best_value = res.fun
                        best_solution = res.x

            if self.evaluations >= self.budget:
                return best_solution

        # Local optimization using hybrid strategy
        remaining_budget = self.budget - self.evaluations
        if remaining_budget > 0:
            res = minimize(func, best_solution, method='trust-constr', bounds=[(lb[i], ub[i]) for i in range(self.dim)],
                           options={'maxiter': remaining_budget})
            if res.fun < best_value:
                best_solution = res.x

        return best_solution