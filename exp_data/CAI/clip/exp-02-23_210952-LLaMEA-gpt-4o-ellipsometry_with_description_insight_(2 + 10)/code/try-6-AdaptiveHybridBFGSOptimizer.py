import numpy as np
from scipy.optimize import minimize

class AdaptiveHybridBFGSOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __adaptive_sampling(self, lower_bounds, upper_bounds, num_samples, iteration, max_iterations):
        """Generates samples with adaptive scaling within the bounds."""
        scaling_factor = 1 - (iteration / max_iterations)
        scaled_bounds = scaling_factor * (upper_bounds - lower_bounds) / 2
        centers = (upper_bounds + lower_bounds) / 2
        return np.random.uniform(centers - scaled_bounds, centers + scaled_bounds, (num_samples, self.dim))
    
    def __bfgs_optimize(self, func, initial_guess, maxiter):
        """Applies BFGS optimization to refine an initial guess."""
        result = minimize(
            func, 
            initial_guess, 
            method='BFGS', 
            options={'maxiter': maxiter}
        )
        return result.x, result.fun

    def __call__(self, func):
        """Optimize the given function using the initialized budget and dimension."""
        # Extract bounds
        lower_bounds = np.array(func.bounds.lb)
        upper_bounds = np.array(func.bounds.ub)
        
        # Define the number of initial samples and iterations for sampling
        num_initial_samples = int(0.2 * self.budget)
        max_iterations = 5
        num_samples_per_iteration = int(num_initial_samples / max_iterations)
        
        best_initial_guess = None
        best_value = float('inf')
        
        # Adaptive sampling across several iterations
        for iteration in range(max_iterations):
            initial_samples = self.__adaptive_sampling(lower_bounds, upper_bounds, num_samples_per_iteration, iteration, max_iterations)
            
            # Evaluate initial samples
            initial_evals = np.array([func(sample) for sample in initial_samples])
            
            # Update the best initial guess
            min_index = np.argmin(initial_evals)
            if initial_evals[min_index] < best_value:
                best_value = initial_evals[min_index]
                best_initial_guess = initial_samples[min_index]
        
        # Adjust budget for BFGS refinement
        remaining_budget = self.budget - num_initial_samples
        
        # Apply BFGS to refine the best initial guess
        best_solution, best_value = self.__bfgs_optimize(
            lambda x: func(np.clip(x, lower_bounds, upper_bounds)), 
            best_initial_guess, 
            maxiter=remaining_budget
        )
        
        return best_solution, best_value