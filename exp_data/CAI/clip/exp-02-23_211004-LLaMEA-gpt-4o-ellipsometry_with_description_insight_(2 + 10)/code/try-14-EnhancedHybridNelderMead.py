import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class EnhancedHybridNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        num_initial_samples = min(10, self.budget // 3)
        
        # Use Sobol sequences for initial sampling to ensure better coverage
        sobol_sampler = qmc.Sobol(d=self.dim, scramble=True)
        initial_points = qmc.scale(sobol_sampler.random(num_initial_samples), bounds[0], bounds[1])
        
        best_solution = None
        best_value = float('inf')
        evaluations = 0
        
        for point in initial_points:
            if evaluations >= self.budget:
                break
            
            def callback(xk):
                nonlocal evaluations
                if evaluations >= self.budget:
                    return True
                return False
            
            # Adjust convergence parameter 'xatol' for better performance
            result = minimize(func, point, method='Nelder-Mead', bounds=bounds.T, callback=callback, options={'xatol': 1e-4})
            evaluations += result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
        
        return best_solution