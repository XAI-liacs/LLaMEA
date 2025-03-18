import numpy as np
from scipy.optimize import minimize

class EnhancedMultiStartNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        samples_per_restart = max(self.budget // 10, 1)
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        
        best_value = float('inf')
        best_solution = None
        
        # Initial dynamic bounds setup
        dynamic_bounds = np.array(bounds)
        
        while self.budget > 0:
            # Narrow down search space dynamically by reducing bounds
            dynamic_bounds[:, 0] = np.maximum(dynamic_bounds[:, 0], best_solution - 0.1 * (dynamic_bounds[:, 1] - dynamic_bounds[:, 0]))
            dynamic_bounds[:, 1] = np.minimum(dynamic_bounds[:, 1], best_solution + 0.1 * (dynamic_bounds[:, 1] - dynamic_bounds[:, 0]))
            
            # Generate random samples within dynamic bounds
            samples = np.random.uniform(dynamic_bounds[:, 0], dynamic_bounds[:, 1], (samples_per_restart, self.dim))
            for sample in samples:
                if self.budget <= 0:
                    break
                # Nelder-Mead Optimization
                result = minimize(func, sample, method='Nelder-Mead', bounds=bounds, options={'maxfev': self.budget, 'adaptive': True})
                
                if result.fun < best_value:
                    best_value = result.fun
                    best_solution = result.x
                    # Reduce remaining budget
                    self.budget -= result.nfev

            # Adaptive restart: Adjust sample size based on remaining budget
            samples_per_restart = max(self.budget // 10, 1)
        
        return best_solution if best_solution is not None else np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)