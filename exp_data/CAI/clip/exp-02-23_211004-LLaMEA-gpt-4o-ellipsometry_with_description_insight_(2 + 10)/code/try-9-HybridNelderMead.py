import numpy as np
from scipy.optimize import minimize

class HybridNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        # Uniformly sample the initial points
        num_initial_samples = min(10, self.budget // 2)
        initial_points = np.random.uniform(bounds[0], bounds[1], (num_initial_samples, self.dim))
        
        best_solution = None
        best_value = float('inf')
        evaluations = 0
        
        for point in initial_points:
            if evaluations >= self.budget:
                break
            # Perform local optimization with BFGS starting from the sampled point
            result = minimize(func, point, method='L-BFGS-B', bounds=bounds.T)
            evaluations += result.nfev

            # Update the best solution found
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
        
        return best_solution