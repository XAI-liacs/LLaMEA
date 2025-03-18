import numpy as np
from scipy.optimize import minimize

class HybridNelderMeadRandomDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        descent_steps = max(self.budget // 20, 1)
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        
        best_value = float('inf')
        best_solution = None
        remaining_budget = self.budget
        
        while remaining_budget > 0:
            # Random starting point
            start_point = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
            
            # Nelder-Mead Optimization
            result = minimize(func, start_point, method='Nelder-Mead', bounds=bounds, options={'maxfev': remaining_budget, 'adaptive': True})
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            remaining_budget -= result.nfev
            
            # Random descent to escape local optima
            for _ in range(descent_steps):
                if remaining_budget <= 0:
                    break
                random_direction = np.random.uniform(-1, 1, self.dim)
                step_size = np.random.uniform(0.01, 0.1)
                new_point = result.x + step_size * random_direction
                
                # Ensure new point is within bounds
                new_point = np.clip(new_point, func.bounds.lb, func.bounds.ub)
                new_value = func(new_point)
                remaining_budget -= 1
                
                if new_value < best_value:
                    best_value = new_value
                    best_solution = new_point

        return best_solution if best_solution is not None else np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)