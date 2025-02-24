import numpy as np
from scipy.optimize import minimize

class AdaptiveHybridBFGSOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __uniform_sampling(self, lower_bounds, upper_bounds, num_samples):
        """Generates uniformly distributed samples within bounds."""
        return np.random.uniform(lower_bounds, upper_bounds, (num_samples, self.dim))
    
    def __adaptive_sampling(self, func, lower_bounds, upper_bounds, num_samples, best_current_solution):
        """Focuses sampling around the best current solution to explore promising regions."""
        exploration_radius = 0.1 * np.ptp(upper_bounds - lower_bounds)
        samples = best_current_solution + np.random.uniform(-exploration_radius, exploration_radius, (num_samples, self.dim))
        return np.clip(samples, lower_bounds, upper_bounds)

    def __bfgs_optimize(self, func, initial_guess, step_size):
        """Applies BFGS optimization with dynamic step size."""
        options = {'maxiter': int(self.budget * step_size), 'gtol': 1e-6}
        result = minimize(func, initial_guess, method='BFGS', options=options)
        return result.x, result.fun

    def __call__(self, func):
        """Optimize the given function using the initialized budget and dimension."""
        # Extract bounds
        lower_bounds = np.array(func.bounds.lb)
        upper_bounds = np.array(func.bounds.ub)

        # Define the number of initial samples to evaluate
        num_initial_samples = int(0.1 * self.budget)
        
        # Generate initial samples
        initial_samples = self.__uniform_sampling(lower_bounds, upper_bounds, num_initial_samples)
        
        # Evaluate initial samples
        initial_evals = np.array([func(sample) for sample in initial_samples])
        
        # Find the best initial sample
        min_index = np.argmin(initial_evals)
        best_initial_guess = initial_samples[min_index]
        best_value = initial_evals[min_index]

        # Set initial step size for BFGS
        step_size = 0.5
        remaining_budget = self.budget - num_initial_samples

        while remaining_budget > 0:
            # Apply adaptive sampling around the best solution found so far
            adaptive_samples = self.__adaptive_sampling(func, lower_bounds, upper_bounds, num_initial_samples, best_initial_guess)
            
            # Evaluate adaptive samples
            adaptive_evals = np.array([func(sample) for sample in adaptive_samples])
            
            # Find the best adaptive sample
            new_min_index = np.argmin(adaptive_evals)
            new_best_guess = adaptive_samples[new_min_index]
            new_best_value = adaptive_evals[new_min_index]
            
            # If a new better solution is found, update best solution and reduce step size
            if new_best_value < best_value:
                best_initial_guess = new_best_guess
                best_value = new_best_value
                step_size = max(step_size / 2, 0.1)
            else:
                # Otherwise, increase step size to explore further
                step_size = min(step_size * 2, 1.0)

            # Adjust budget for BFGS refinement
            bfgs_budget = int(remaining_budget * step_size)
            remaining_budget -= bfgs_budget

            # Apply BFGS to refine the best guess
            best_initial_guess, best_value = self.__bfgs_optimize(lambda x: func(np.clip(x, lower_bounds, upper_bounds)), best_initial_guess, step_size)

        return best_initial_guess, best_value