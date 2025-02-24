import numpy as np
from scipy.optimize import minimize

class EnhancedHybridNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        num_initial_samples = min(10, self.budget // 3)  # Allocate more budget for exploitation
        initial_points = np.random.uniform(bounds[0], bounds[1], (num_initial_samples, self.dim))
        
        best_solution = None
        best_value = float('inf')
        evaluations = 0
        
        for point in initial_points:
            if evaluations >= self.budget:
                break
            # Adaptive Nelder-Mead: dynamically adjust options for convergence
            options = {'adaptive': True, 'xatol': 1e-4, 'fatol': 1e-4}
            result = minimize(func, point, method='Nelder-Mead', options=options)
            evaluations += result.nfev

            # Update the best solution found
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
        
        # Further refinement with remaining budget
        remaining_budget = self.budget - evaluations
        if remaining_budget > 0 and best_solution is not None:
            options = {'adaptive': True, 'maxiter': remaining_budget, 'xatol': 1e-5, 'fatol': 1e-5}
            result = minimize(func, best_solution, method='Nelder-Mead', options=options)
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        return best_solution