import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

class HybridEvolutionaryLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')

        # Differential Evolution for initial exploration
        differential_budget = max(10, self.budget // 2)
        bounds = [(lb[i], ub[i]) for i in range(self.dim)]
        result = differential_evolution(func, bounds, maxiter=differential_budget, disp=False)
        self.budget -= result.nfev

        if result.fun < best_value:
            best_value = result.fun
            best_solution = result.x

        # Adaptive local optimization using BFGS
        options = {'maxiter': self.budget, 'disp': False}
        def callback(xk):
            nonlocal best_value, best_solution
            current_value = func(xk)
            if current_value < best_value:
                best_value = current_value
                best_solution = xk

        # Refining bounds around the best solution
        bounds = [(max(lb[i], best_solution[i] - 0.1*(ub[i]-lb[i])), 
                   min(ub[i], best_solution[i] + 0.1*(ub[i]-lb[i]))) for i in range(self.dim)]

        result = minimize(func, best_solution, method='L-BFGS-B', bounds=bounds, options=options, callback=callback)

        return best_solution if result.success else result.x