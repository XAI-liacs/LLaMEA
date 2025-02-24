import numpy as np
from scipy.optimize import minimize, differential_evolution

class NovelMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Define bounds from the function's bounds
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Step 1: Global Exploration using Differential Evolution
        def track_evaluations(x):
            if self.evaluations < self.budget:
                self.evaluations += 1
                return func(x)
            else:
                raise Exception("Exceeded budget of function evaluations")

        de_bounds = list(zip(lb, ub))
        de_result = differential_evolution(track_evaluations, de_bounds, strategy='best1bin', maxiter=int(self.budget/2), popsize=15, polish=False)
        
        # Store the best solution found
        best_solution = de_result.x
        best_objective = de_result.fun

        # Step 2: Local Optimization from DE result
        if self.evaluations < self.budget:
            result = minimize(track_evaluations, best_solution, method='L-BFGS-B', bounds=de_bounds)
            best_solution = result.x
            best_objective = result.fun

        return best_solution