import numpy as np
from scipy.optimize import minimize

class ABC_LO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        # Initialize bounds from the function
        bounds = [(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)]
        
        # Uniformly sample initial solutions
        weights = np.random.uniform(0.1, 1.0, self.dim)  
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
        
        # Refine initial guess by averaging with previous result for convergence improvement
        refined_initial_guess = (result.x + initial_guess + np.random.uniform(-0.05, 0.05, self.dim) * 0.1) / 2  # Change made here
        
        # Refine solution with adjusted bounds
        refined_result = minimize(func, refined_initial_guess, method='L-BFGS-B', bounds=new_bounds, options=options)
        
        # Return the best found solution
        return refined_result.x, refined_result.fun