import numpy as np
from scipy.optimize import minimize

class ABC_LO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        # Initialize bounds from the function
        bounds = [(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)]
        
        # Dynamically adjust weights based on iteration
        weights = np.linspace(0.5, 1.5, self.dim)  # Adjusted weights
        initial_guess = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim) * weights
        
        # Local optimization using BFGS within the bounds
        options = {'maxiter': self.budget, 'disp': False}
        result = minimize(func, initial_guess, method='L-BFGS-B', bounds=bounds, options=options)
        
        # Adjust bounds based on the optimization result
        new_bounds = []
        for i in range(self.dim):
            center = result.x[i]
            width = (bounds[i][1] - bounds[i][0]) * 0.5
            new_bounds.append((max(center - width, func.bounds.lb[i]), min(center + width, func.bounds.ub[i])))
        
        # Refine solution with adjusted bounds and dynamic weights
        refined_weights = np.linspace(0.8, 1.2, self.dim)  # Adjusted weights for refinement
        refined_result = minimize(func, result.x * refined_weights, method='L-BFGS-B', bounds=new_bounds, options=options)
        
        # Return the best found solution
        return refined_result.x, refined_result.fun