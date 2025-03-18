import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Extract bounds from the function
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        
        # Initial exploration with adaptive sampling
        num_samples = min(15, self.budget // 3)  # Slightly increased initial samples and refined ratio
        samples = np.random.uniform(low=[b[0] for b in bounds],
                                    high=[b[1] for b in bounds],
                                    size=(num_samples, self.dim))
        
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
        
        # Apply BFGS starting from the best initial sample
        result = minimize(count_calls, best_sample, method='L-BFGS-B', bounds=bounds)
        
        # Return the best found solution and its cost
        return result.x, result.fun