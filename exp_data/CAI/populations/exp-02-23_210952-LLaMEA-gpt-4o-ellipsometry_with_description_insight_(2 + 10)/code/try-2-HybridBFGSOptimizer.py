import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class HybridBFGSOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __uniform_sampling(self, lower_bounds, upper_bounds, num_samples):
        """Generates uniformly distributed samples within bounds."""
        return np.random.uniform(lower_bounds, upper_bounds, (num_samples, self.dim))
    
    def __sobol_sampling(self, lower_bounds, upper_bounds, num_samples):
        """Generates Sobol sequence samples within bounds."""
        sampler = Sobol(d=self.dim, scramble=True)
        samples = sampler.random_base2(m=int(np.log2(num_samples)))
        return lower_bounds + (upper_bounds - lower_bounds) * samples

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
        
        # Generate initial samples
        initial_samples = self.__sobol_sampling(lower_bounds, upper_bounds, num_initial_samples)
        
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