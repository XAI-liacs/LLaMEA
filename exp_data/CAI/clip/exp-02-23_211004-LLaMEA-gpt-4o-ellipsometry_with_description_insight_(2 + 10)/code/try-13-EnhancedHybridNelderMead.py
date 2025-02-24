import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class EnhancedHybridNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        # Adaptively determine the number of initial samples based on the budget
        num_initial_samples = min(10, self.budget // 3)
        sampler = Sobol(d=self.dim, scramble=True)
        initial_points = sampler.random_base2(m=int(np.log2(num_initial_samples)))
        initial_points = bounds[0] + initial_points * (bounds[1] - bounds[0])
        
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
            result = minimize(func, point, method='Nelder-Mead', bounds=bounds.T, callback=callback)
            evaluations += result.nfev

            # Update the best solution found
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
        
        return best_solution