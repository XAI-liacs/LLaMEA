import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Extract bounds from the function
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        
        # Initial exploration with Sobol sequence sampling (adaptive sample size)
        num_samples = min(20, self.budget // 3)  # Adjusted sample size
        sampler = Sobol(d=self.dim, scramble=True)
        samples = sampler.random(n=num_samples) * (np.array([b[1] for b in bounds]) - np.array([b[0] for b in bounds])) + np.array([b[0] for b in bounds])
        
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
        
        # Apply L-BFGS-B starting from the best initial sample (method adjustment)
        result = minimize(count_calls, best_sample, method='L-BFGS-B', bounds=bounds)
        
        # Return the best found solution and its cost
        return result.x, result.fun