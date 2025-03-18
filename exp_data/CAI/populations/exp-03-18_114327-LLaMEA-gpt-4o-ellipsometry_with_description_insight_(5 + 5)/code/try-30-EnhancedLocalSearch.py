import numpy as np
from scipy.optimize import minimize

class EnhancedLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def _evaluate(self, func, x):
        if self.evals >= self.budget:
            raise RuntimeError("Budget exceeded")
        self.evals += 1
        return func(x)

    def _initialize_population(self, bounds, num_samples):
        return np.random.uniform(bounds.lb, bounds.ub, (num_samples, self.dim))

    def __call__(self, func):
        bounds = func.bounds
        num_initial_samples = min(10, max(3, self.budget // 4))  # Adjusted initial samples
        population = self._initialize_population(bounds, num_initial_samples)
        
        evaluated_points = [(x, self._evaluate(func, x)) for x in population]
        evaluated_points.sort(key=lambda x: x[1])
        best_x, best_val = evaluated_points[0]

        # Take top 4 samples for local exploitation
        for x, _ in evaluated_points[:4]:  
            result = minimize(lambda x: self._evaluate(func, x), x, 
                              method='L-BFGS-B', bounds=[(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)])
            
            if result.fun < best_val:
                best_x, best_val = result.x, result.fun

        # Refine bounds based on best solution
        new_bounds_lb = np.maximum(bounds.lb, best_x - 0.1 * (bounds.ub - bounds.lb))
        new_bounds_ub = np.minimum(bounds.ub, best_x + 0.1 * (bounds.ub - bounds.lb))
        refined_bounds = [(lb, ub) for lb, ub in zip(new_bounds_lb, new_bounds_ub)]

        # Final local refinement
        result = minimize(lambda x: self._evaluate(func, x), best_x, 
                          method='BFGS', bounds=refined_bounds)

        return result.x