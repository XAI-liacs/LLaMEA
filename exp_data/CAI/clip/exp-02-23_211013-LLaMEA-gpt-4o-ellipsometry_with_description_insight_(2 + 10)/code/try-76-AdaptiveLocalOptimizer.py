import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class AdaptiveLocalOptimizer:
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
        no_improvement_count = 0  # Line added
        
        # Uniformly sample initial points to cover the parameter space
        sobol_engine = Sobol(d=self.dim, scramble=True)
        initial_points = sobol_engine.random_base2(m=4) * (ub - lb) + lb
        
        for point in initial_points:
            if current_budget >= self.budget or no_improvement_count >= 3:  # Line changed
                break

            # Define a local optimization procedure
            res = minimize(func, point, method='L-BFGS-B', bounds=zip(lb, ub))
            current_budget += res.nfev  # Number of function evaluations
            
            # Update the best solution found
            if res.fun < best_score:
                best_solution = res.x
                best_score = res.fun
                no_improvement_count = 0  # Reset counter if improvement is found
            else:
                no_improvement_count += 1  # Increment counter if no improvement
            
            # Dynamically adjust bounds and constraints if needed
            # Here, we can add logic to refine the search space if the budget allows
            # For simplicity, this step is omitted but can be implemented based on specific requirements

        return best_solution