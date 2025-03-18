import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Define the bounds
        lb = func.bounds.lb
        ub = func.bounds.ub
        
        # Initialize the best solution as None
        best_solution = None
        best_value = float('inf')
        
        # Calculate the number of initial samples, a fraction of the total budget
        num_samples = max(1, self.budget // 10)
        
        # Generate uniform random initial samples
        initial_samples = np.random.uniform(lb, ub, (num_samples, self.dim))
        remaining_budget = self.budget - num_samples
        
        # Evaluate initial samples and store the best
        for sample in initial_samples:
            value = func(sample)
            if value < best_value:
                best_solution = sample
                best_value = value
            remaining_budget -= 1
        
        # Define a local optimization function
        def local_optimization(starting_point):
            res = minimize(func, starting_point, method='L-BFGS-B', bounds=list(zip(lb, ub)))
            return res.x, res.fun
        
        # Perform local optimization from the best initial sample
        if remaining_budget > 0:
            solution, value = local_optimization(best_solution)
            if value < best_value:
                best_solution = solution
                best_value = value
        
        return best_solution