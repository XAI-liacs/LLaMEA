import numpy as np
from scipy.optimize import minimize

class AdaptiveNelderMeadDynamicRestart:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)])
        initial_simplex = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.dim + 1, self.dim))
        
        def wrapped_func(x):
            return func(np.clip(x, bounds[:, 0], bounds[:, 1]))
        
        best_value = float('inf')
        best_solution = None
        
        while self.budget > 0:
            # Perform Nelder-Mead optimization with the current budget
            result = minimize(wrapped_func, initial_simplex[0], method='Nelder-Mead', 
                              options={'maxfev': self.budget, 'initial_simplex': initial_simplex})
            
            self.budget -= result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_solution = np.clip(result.x, bounds[:, 0], bounds[:, 1])
            
            # Detect convergence and decide if a restart is needed
            if result.success or self.budget <= 0:
                break
            
            # Dynamic restart by generating a new simplex around the best solution found
            initial_simplex = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.dim + 1, self.dim))
            initial_simplex[0] = best_solution
        
        return best_solution