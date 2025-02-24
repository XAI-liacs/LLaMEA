import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

class AdvancedNaturalComputingOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        bounds = list(zip(lb, ub))
        remaining_budget = self.budget

        # Differential Evolution for global search
        def de_obj(x):
            nonlocal remaining_budget
            if remaining_budget <= 0:
                return np.inf
            remaining_budget -= 1
            return func(x)

        result_de = differential_evolution(de_obj, bounds, maxiter=remaining_budget//(self.dim*10), disp=False)
        best_solution = result_de.x
        best_value = result_de.fun

        # Adaptive local refinement using Nelder-Mead
        if remaining_budget > 0:
            result_nm = minimize(func, best_solution, method='Nelder-Mead', options={'maxfev': remaining_budget, 'disp': False})
            if result_nm.fun < best_value:
                best_value = result_nm.fun
                best_solution = result_nm.x

        return best_solution