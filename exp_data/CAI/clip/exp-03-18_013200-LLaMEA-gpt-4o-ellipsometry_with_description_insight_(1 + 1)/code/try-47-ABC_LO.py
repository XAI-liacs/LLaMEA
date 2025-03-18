import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class ABC_LO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        # Initialize bounds from the function
        bounds = [(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)]
        
        # Uniformly sample initial solutions
        sampler = Sobol(d=self.dim, scramble=True)
        weights = sampler.random_base2(m=1)[0] * (func.bounds.ub - func.bounds.lb) + func.bounds.lb
        initial_guess = weights
        
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
        refined_initial_guess = (result.x + initial_guess + np.random.uniform(-0.05, 0.05, self.dim)) / 2
        
        # Refine solution with adjusted bounds
        refined_result = minimize(func, refined_initial_guess, method='L-BFGS-B', bounds=new_bounds, options=options)
        
        # Return the best found solution
        return refined_result.x, refined_result.fun