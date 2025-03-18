import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Extract bounds from the function
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        
        # Initial exploration with uniform sampling
        num_samples = min(15, self.budget // 3)  # More samples with a third of the budget
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
        
        # Dynamic adjustment of bounds based on best sample
        adaptive_bounds = [(max(b[0], best_sample[i] - 0.1 * (b[1] - b[0])), 
                            min(b[1], best_sample[i] + 0.1 * (b[1] - b[0]))) for i, b in enumerate(bounds)]
        
        # Define a wrapper for the cost function to count evaluations
        self.evaluation_count = 0
        
        def count_calls(x):
            if self.evaluation_count < remaining_budget:
                self.evaluation_count += 1
                return func(x)
            else:
                raise RuntimeError("Exceeded budget in local optimization")
        
        # Apply BFGS starting from the best initial sample with adaptive bounds
        result = minimize(count_calls, best_sample, method='L-BFGS-B', bounds=adaptive_bounds)
        
        # Return the best found solution and its cost
        return result.x, result.fun