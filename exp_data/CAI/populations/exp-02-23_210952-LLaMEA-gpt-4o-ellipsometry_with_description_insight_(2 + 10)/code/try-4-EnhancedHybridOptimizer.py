import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __latin_hypercube_sampling(self, lower_bounds, upper_bounds, num_samples):
        """Generates Latin Hypercube samples within bounds."""
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=num_samples)
        scaled_samples = qmc.scale(sample, lower_bounds, upper_bounds)
        return scaled_samples
    
    def __bfgs_optimize(self, func, initial_guess):
        """Applies BFGS optimization to refine an initial guess."""
        result = minimize(func, initial_guess, method='BFGS', options={'maxiter': self.budget})
        return result.x, result.fun

    def __call__(self, func):
        """Optimize the given function using the initialized budget and dimension."""
        # Extract bounds
        lower_bounds = np.array(func.bounds.lb)
        upper_bounds = np.array(func.bounds.ub)
        
        # Define the number of initial samples to evaluate
        num_initial_samples = int(0.2 * self.budget)
        
        # Generate initial samples using Latin Hypercube Sampling
        initial_samples = self.__latin_hypercube_sampling(lower_bounds, upper_bounds, num_initial_samples)
        
        # Evaluate initial samples
        initial_evals = np.array([func(sample) for sample in initial_samples])
        
        # Find the best initial sample
        min_index = np.argmin(initial_evals)
        best_initial_guess = initial_samples[min_index]
        
        # Adjust budget for BFGS refinement
        remaining_budget = self.budget - num_initial_samples
        
        # Apply BFGS to refine the best initial guess
        best_solution, best_value = self.__bfgs_optimize(lambda x: func(np.clip(x, lower_bounds, upper_bounds)), best_initial_guess)
        
        return best_solution, best_value