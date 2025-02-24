import numpy as np
from scipy.optimize import minimize

class AdaptiveMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        
        # Step 1: Adaptive Sampling for Initial Guesses
        num_initial_samples = min(5, self.budget // 2)  # Use half of budget for initial sampling
        initial_samples = np.random.uniform(
            low=func.bounds.lb, 
            high=func.bounds.ub, 
            size=(num_initial_samples, self.dim)
        )
        
        # Evaluate initial samples
        initial_evaluations = [func(x) for x in initial_samples]
        remaining_budget = self.budget - num_initial_samples
        
        # Step 2: Perform a local optimization from the best initial sample
        best_initial_idx = np.argmin(initial_evaluations)
        best_initial_sample = initial_samples[best_initial_idx]
        
        # Adaptive constraint tightening factor
        tighten_factor = 0.9
        
        # Define the objective function for the local optimizer
        def local_objective(x):
            return func(x)
        
        # Use L-BFGS-B for local optimization with dynamic bounds tightening
        reduced_bounds = [(lb + tighten_factor * (ub - lb), ub - tighten_factor * (ub - lb)) 
                          for lb, ub in bounds]
        
        result = minimize(
            local_objective, 
            best_initial_sample, 
            method='L-BFGS-B',
            bounds=reduced_bounds,
            options={'maxiter': remaining_budget}
        )
        
        return result.x

# Example of usage:
# Assume func is a black-box function with attributes bounds.lb and bounds.ub
# optimizer = AdaptiveMetaheuristicOptimizer(budget=100, dim=2)
# best_parameters = optimizer(func)