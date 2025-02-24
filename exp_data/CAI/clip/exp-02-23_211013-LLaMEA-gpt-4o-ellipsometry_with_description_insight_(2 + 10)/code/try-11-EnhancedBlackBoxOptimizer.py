import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

class EnhancedBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        func_calls = 0
        # Extract bounds
        lb = func.bounds.lb
        ub = func.bounds.ub
        bounds = [(l, u) for l, u in zip(lb, ub)]

        # Define the function to optimize
        def objective(x):
            nonlocal func_calls
            if func_calls >= self.budget:
                return float('inf')
            func_calls += 1
            return func(x)

        # Use Differential Evolution for global exploration
        def de_objective(x):
            return objective(x)

        result_de = differential_evolution(de_objective, bounds, maxiter=int(self.budget * 0.5), polish=False, disp=False)

        # Perform local optimization using Nelder-Mead method from best DE result
        result_nm = minimize(objective, result_de.x, method='Nelder-Mead', bounds=bounds)

        # Return best solution found
        if result_nm.success:
            return result_nm.x
        else:
            return result_de.x