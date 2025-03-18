import numpy as np
from scipy.optimize import minimize

class AdaptiveNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        # Start with a uniform sampling to get initial guess
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        initial_guess = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)  # Change: varying initial guess range

        # Define the callback function to track evaluation budget
        def callback(xk):
            self.eval_count += 1
            if self.eval_count >= self.budget:
                raise StopIteration("Budget exhausted")
        
        # Define a function to wrap the original function with budget control
        def func_with_budget(x):
            if self.eval_count >= self.budget:
                raise StopIteration("Budget exhausted")
            self.eval_count += 1
            return func(x)
        
        self.eval_count = 0
        best_result = None
        
        try:
            # Use Nelder-Mead with adaptive bounds
            result = minimize(
                func_with_budget, 
                initial_guess, 
                method='Nelder-Mead', 
                bounds=bounds,
                callback=callback,
                options={'maxiter': self.budget, 'adaptive': True, 'xatol': 1e-8}
            )
            best_result = result
        except StopIteration:
            pass
        
        # Introduce a restart mechanism if budget allows
        if self.eval_count < self.budget:
            new_initial_guess = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
            try:
                result = minimize(
                    func_with_budget, 
                    new_initial_guess, 
                    method='Nelder-Mead', 
                    bounds=bounds,
                    callback=callback,
                    options={'maxiter': self.budget - self.eval_count, 'adaptive': True, 'xatol': 1e-8}
                )
                if not best_result or result.fun < best_result.fun:
                    best_result = result
            except StopIteration:
                pass
        
        # Refine bounds and iteratively improve
        if self.eval_count < self.budget:
            refined_bounds = [(max(func.bounds.lb[i], best_result.x[i] - 0.1 * (func.bounds.ub[i] - func.bounds.lb[i])), 
                               min(func.bounds.ub[i], best_result.x[i] + 0.1 * (func.bounds.ub[i] - func.bounds.lb[i])))
                              for i in range(self.dim)]
            try:
                result = minimize(
                    func_with_budget, 
                    best_result.x, 
                    method='Nelder-Mead', 
                    bounds=refined_bounds,
                    callback=callback,
                    options={'maxiter': self.budget - self.eval_count, 'adaptive': True, 'xatol': 1e-6}
                )
                if result.fun < best_result.fun:
                    best_result = result
            except StopIteration:
                pass
        
        return best_result.x