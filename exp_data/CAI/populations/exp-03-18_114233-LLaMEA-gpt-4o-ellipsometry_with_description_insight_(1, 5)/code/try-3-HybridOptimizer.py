import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Define the bounds based on the function's bounds
        bounds = func.bounds
        lb = bounds.lb
        ub = bounds.ub
        
        # Use grid sampling for initial exploration across the parameter space
        num_samples = min(10, self.budget // 2)  # Take up to 10 initial samples or half the budget
        grids = [np.linspace(lb[i], ub[i], num_samples) for i in range(self.dim)]
        samples = np.array(np.meshgrid(*grids)).T.reshape(-1, self.dim)
        
        # Evaluate initial samples and store the best
        best_sample = None
        best_value = float('inf')
        evaluations = 0
        
        for sample in samples:
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_sample = sample
        
        # Use BFGS for local optimization starting from the best sample found
        result = minimize(func, best_sample, bounds=list(zip(lb, ub)), method='L-BFGS-B', options={'maxfun': self.budget - evaluations})
        return result.x

# Usage example (uncomment to run in an appropriate environment):
# def func(x):
#     return (x[0] - 2)**2 + (x[1] - 3)**2
# func.bounds = lambda: None
# func.bounds.lb = np.array([0, 0])
# func.bounds.ub = np.array([5, 5])
# optimizer = HybridOptimizer(budget=50, dim=2)
# best_params = optimizer(func)
# print("Best parameters:", best_params)