import numpy as np
from scipy.optimize import minimize

class AdaptiveGaussianMutator:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def __call__(self, func):
        bounds = np.array(list(zip(func.bounds.lb, func.bounds.ub)))
        num_samples = min(10, self.budget // 2)
        
        # Step 1: Uniform sampling for initial exploration
        initial_samples = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_samples, self.dim))
        initial_values = [func(x) for x in initial_samples]
        self.evals += num_samples
        
        # Step 2: Select the best initial guess and refine with local optimizer
        best_idx = np.argmin(initial_values)
        best_guess = initial_samples[best_idx]
        
        # Define the bounds again for the local optimizer
        local_bounds = [(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)]
        
        # Local optimizer using L-BFGS-B
        result = minimize(
            func,
            best_guess,
            method='L-BFGS-B',
            bounds=local_bounds,
            options={'maxfun': self.budget - self.evals}
        )
        
        # Update the number of evaluations used by the optimizer
        self.evals += result.nfev

        # If evaluations are not yet exhausted, perform Adaptive Gaussian Mutation
        while self.evals < self.budget:
            # Generate a Gaussian noise centered at the best solution found
            mutation_std_dev = 0.05 * (bounds[:, 1] - bounds[:, 0])
            gaussian_mutation = np.random.normal(0, mutation_std_dev, self.dim)
            new_guess = np.clip(result.x + gaussian_mutation, bounds[:, 0], bounds[:, 1])
            
            # Evaluate the new guess
            new_value = func(new_guess)
            self.evals += 1
            
            # If the new guess is better, update the result
            if new_value < result.fun:
                result.x = new_guess
                result.fun = new_value
        
        return result.x