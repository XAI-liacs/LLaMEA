import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

class HybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Define the search space
        lb = func.bounds.lb
        ub = func.bounds.ub
        
        # Initialize variables
        current_budget = 0
        best_solution = None
        best_score = float('inf')
        
        def budget_constraint(x):
            nonlocal current_budget
            if current_budget >= self.budget:
                raise Exception("Budget exceeded")
            current_budget += 1
            return func(x)

        # Define Differential Evolution as global optimizer
        result = differential_evolution(budget_constraint, bounds=zip(lb, ub), strategy='best1bin', disp=False)
        current_budget += result.nfev  # Number of function evaluations

        if result.success:
            # Use L-BFGS-B for local refinement of the best DE solution
            res = minimize(budget_constraint, result.x, method='L-BFGS-B', bounds=zip(lb, ub))
            current_budget += res.nfev  # Number of function evaluations
            
            if res.fun < best_score:
                best_solution = res.x
                best_score = res.fun
        
        return best_solution