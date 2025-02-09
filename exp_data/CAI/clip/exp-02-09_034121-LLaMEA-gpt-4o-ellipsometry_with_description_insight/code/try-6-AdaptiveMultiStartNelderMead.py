import numpy as np
from scipy.optimize import minimize

class AdaptiveMultiStartNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        initial_samples = 5  # Initial number of random samples
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        
        best_value = float('inf')
        best_solution = None
        utilized_budget = 0
        
        while self.budget > utilized_budget:
            # Determine number of samples based on remaining budget
            samples_per_restart = max((self.budget - utilized_budget) // initial_samples, 1)
            samples = np.random.uniform(func.bounds.lb, func.bounds.ub, (samples_per_restart, self.dim))
            
            for sample in samples:
                if utilized_budget >= self.budget:
                    break

                # Nelder-Mead Optimization with adaptive budget allocation
                remaining_budget = self.budget - utilized_budget
                options = {'maxfev': remaining_budget, 'adaptive': True}
                result = minimize(func, sample, method='Nelder-Mead', bounds=bounds, options=options)
                
                # Budget utilization logging
                utilized_budget += result.nfev

                # Update best solution found
                if result.fun < best_value:
                    best_value = result.fun
                    best_solution = result.x
                
                # Dynamic budget reallocation based on performance
                if result.success and result.fun < best_value:
                    initial_samples = max(initial_samples - 1, 1)
                else:
                    initial_samples += 1
        
        return best_solution if best_solution is not None else np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)