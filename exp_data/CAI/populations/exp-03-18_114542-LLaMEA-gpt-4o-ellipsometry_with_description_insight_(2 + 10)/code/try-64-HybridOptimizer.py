import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Halton

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Extract bounds from the function
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        
        # Dynamic initial sample size based on dimension and budget
        num_samples = min(self.dim * 2, max(3, self.budget // 5))  # Adjusted sample size
        
        # Generate initial samples using Halton sequence
        sampler = Halton(d=self.dim, scramble=True)
        samples = sampler.random(n=num_samples)
        samples = samples * (np.array([b[1] for b in bounds]) - np.array([b[0] for b in bounds])) + np.array([b[0] for b in bounds])

        # Evaluate initial samples
        evaluated_samples = [(x, func(x)) for x in samples]
        
        # Sort samples based on their evaluated cost
        evaluated_samples.sort(key=lambda item: item[1])
        
        # Start with the best sample
        best_sample, best_cost = evaluated_samples[0]
        
        # Remaining budget for the local optimization
        remaining_budget = self.budget - num_samples
        
        # Define a wrapper for the cost function to count evaluations
        self.evaluation_count = 0
        
        def count_calls(x):
            if self.evaluation_count < remaining_budget:
                self.evaluation_count += 1
                return func(x)
            else:
                raise RuntimeError("Exceeded budget in local optimization")
        
        # Apply BFGS starting from the best initial sample with refined termination
        result = minimize(count_calls, best_sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': remaining_budget // 2, 'ftol': 1e-9}) # Adjusted maxiter and termination
        
        # Return the best found solution and its cost
        return result.x, result.fun