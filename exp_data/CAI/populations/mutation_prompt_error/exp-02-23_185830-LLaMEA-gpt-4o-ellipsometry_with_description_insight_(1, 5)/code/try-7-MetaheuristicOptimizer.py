import numpy as np
from scipy.optimize import minimize

class MetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        
        # Step 1: Uniform Sampling for Initial Guesses
        num_initial_samples = min(5, self.budget // 2)  # Use half of budget for initial sampling
        initial_samples = np.random.uniform(
            low=func.bounds.lb, 
            high=func.bounds.ub, 
            size=(num_initial_samples, self.dim)
        )
        
        # Evaluate initial samples
        initial_evaluations = [func(x) for x in initial_samples]
        remaining_budget = self.budget - num_initial_samples
        
        # Step 2: Local Optimization and potential random restart
        best_initial_idx = np.argmin(initial_evaluations)
        best_initial_sample = initial_samples[best_initial_idx]
        
        # Define the objective function for the local optimizer
        def local_objective(x):
            return func(x)
        
        # Use BFGS or Nelder-Mead based on performance for local optimization
        optimization_method = 'BFGS' if initial_evaluations[best_initial_idx] < 0.5 else 'Nelder-Mead'  # Line modified
        result = minimize(
            local_objective, 
            best_initial_sample, 
            method=optimization_method,
            bounds=bounds,
            options={'maxiter': remaining_budget}
        )
        
        # Random restart if necessary
        if result.fun > np.min(initial_evaluations): 
            random_idx = np.random.choice(num_initial_samples)
            result = minimize(
                local_objective, 
                initial_samples[random_idx], 
                method=optimization_method,
                bounds=bounds,
                options={'maxiter': remaining_budget}
            )
        
        return result.x

# Example of usage:
# Assume func is a black-box function with attributes bounds.lb and bounds.ub
# optimizer = MetaheuristicOptimizer(budget=100, dim=2)
# best_parameters = optimizer(func)