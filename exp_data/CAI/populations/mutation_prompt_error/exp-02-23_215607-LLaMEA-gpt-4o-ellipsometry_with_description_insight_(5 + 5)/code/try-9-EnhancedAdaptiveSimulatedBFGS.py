import numpy as np
from scipy.optimize import minimize

class EnhancedAdaptiveSimulatedBFGS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initial sampling across the parameter space
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        initial_samples = np.random.uniform(bounds[:, 0], bounds[:, 1], (10, self.dim))
        
        best_solution = None
        best_value = float('inf')
        remaining_budget = self.budget
        
        # Evaluate initial samples
        for sample in initial_samples:
            if remaining_budget <= 0:
                break
            remaining_budget -= 1
            value = func(sample)
            if value < best_value:
                best_value = value
                best_solution = sample

        def local_search(x0, bounds, remaining_budget, method='L-BFGS-B'):
            nonlocal best_solution, best_value
            if remaining_budget <= 0:
                return
            res = minimize(func, x0, method=method, bounds=bounds, options={'maxfun': remaining_budget})
            used_budget = res.nfev
            remaining_budget -= used_budget
            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x
        
        # Iteratively refine the search space
        while remaining_budget > 0:
            # Adjust bounds around the best solution more aggressively
            aggressive_factor = 0.05
            adjusted_bounds = np.clip(np.array([best_solution - aggressive_factor*(func.bounds.ub-func.bounds.lb), 
                                                best_solution + aggressive_factor*(func.bounds.ub-func.bounds.lb)]).T, 
                                      func.bounds.lb, func.bounds.ub)
            # Use derivative-free method if budget is low
            if remaining_budget < 0.5 * self.budget:
                local_search(best_solution, adjusted_bounds, remaining_budget, method='Nelder-Mead')
            else:
                # Local optimization around current best solution
                local_search(best_solution, adjusted_bounds, remaining_budget)

        return best_solution