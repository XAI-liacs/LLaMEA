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
        initial_guess = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        
        # Local optimization using BFGS within the bounds
        options = {'maxiter': self.budget, 'disp': False}
        # Modify to include a slight perturbation for better exploration
        perturbed_guess = initial_guess + np.random.normal(0, 0.01, self.dim)
        result = minimize(func, perturbed_guess, method='L-BFGS-B', bounds=bounds, options=options)
        
        # Adjust bounds based on the optimization result
        new_bounds = []
        for i in range(self.dim):
            center = result.x[i]
            width = (bounds[i][1] - bounds[i][0]) * 0.5
            new_bounds.append((max(center - width, func.bounds.lb[i]), min(center + width, func.bounds.ub[i])))

        # Refine solution with adjusted bounds
        refined_result = minimize(func, result.x, method='L-BFGS-B', bounds=new_bounds, options=options)
        
        # Return the best found solution
        return refined_result.x, refined_result.fun