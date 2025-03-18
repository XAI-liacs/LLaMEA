import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        num_initial_guesses = min(self.budget // 8, 12)  # Adjusted number of initial guesses
        
        # Adaptive sampling based on variance within bounds
        initial_guesses = np.array([np.random.uniform(low=b[0], high=b[1], size=self.dim) 
                                    for b in [bounds] * num_initial_guesses])

        best_solution = None
        best_value = float('inf')
        evaluations = 0

        for guess in initial_guesses:
            if evaluations >= self.budget:
                break
            
            # Use dual-phase local search: first BFGS, then Nelder-Mead if needed
            result = minimize(func, guess, method='L-BFGS-B', bounds=bounds, 
                              options={'maxfun': (self.budget - evaluations) // 2})
            evaluations += result.nfev

            if evaluations < self.budget:
                result_nm = minimize(func, result.x, method='Nelder-Mead', 
                                     options={'maxfev': self.budget - evaluations})
                evaluations += result_nm.nfev

                if result_nm.fun < result.fun:
                    result = result_nm

            # Update the best solution found so far
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        return best_solution