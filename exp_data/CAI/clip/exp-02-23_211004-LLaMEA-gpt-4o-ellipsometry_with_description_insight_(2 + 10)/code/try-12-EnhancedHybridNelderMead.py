import numpy as np
from scipy.optimize import minimize

class EnhancedHybridNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        # Adaptively determine the number of initial samples based on the budget
        num_initial_samples = min(15, self.budget // 3)  # Adjusted initial sample size for better exploration
        initial_points = np.random.uniform(bounds[0], bounds[1], (num_initial_samples, self.dim))
        
        best_solution = None
        best_value = float('inf')
        evaluations = 0
        
        for point in initial_points:
            if evaluations >= self.budget:
                break
            
            # Define a callback function for early stopping
            def callback(xk):
                nonlocal evaluations
                if evaluations >= self.budget:
                    return True
                return False
            
            # Perform local optimization with Nelder-Mead with early stopping
            result = minimize(func, point, method='Nelder-Mead', options={'maxiter': self.budget-evaluations}, callback=callback)  # Adjusted maxiter option
            evaluations += result.nfev

            # Update the best solution found
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
        
        return best_solution