import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

class DE_LBFGSB_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        
        # Function to clip parameters within bounds
        def bounded_func(x):
            x_clipped = np.clip(x, func.bounds.lb, func.bounds.ub)
            return func(x_clipped)
        
        # Calculate budget split between DE and L-BFGS-B
        de_budget = int(0.7 * self.budget)
        lbfgs_budget = self.budget - de_budget

        # Differential Evolution for global search
        de_result = differential_evolution(
            bounded_func, bounds, 
            maxiter=de_budget // (self.dim + 1), 
            polish=False, disp=False
        )

        # Use L-BFGS-B for local refinement
        options = {'maxiter': lbfgs_budget}
        lbfgs_result = minimize(
            bounded_func, de_result.x, method='L-BFGS-B', bounds=bounds, options=options
        )

        return lbfgs_result.x

# Example usage:
# optimizer = DE_LBFGSB_Optimizer(budget=100, dim=2)
# best_solution = optimizer(your_black_box_function)