import numpy as np
from scipy.optimize import minimize

class AdaptiveLocalSearch:
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
        num_initial_samples = min(10, self.budget // 2)
        population = self._initialize_population(bounds, num_initial_samples)
        
        # Evaluate initial samples
        evaluated_points = [(x, self._evaluate(func, x)) for x in population]
        
        # Sort by function value
        evaluated_points.sort(key=lambda x: x[1])
        best_x, best_val = evaluated_points[0]

        # Use BFGS to refine the best solutions
        for x, _ in evaluated_points[:5]:  # Take top 5 samples for local exploitation
            result = minimize(lambda x: self._evaluate(func, x), x, 
                              method='L-BFGS-B', bounds=[(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)])
            
            if result.fun < best_val:
                best_x, best_val = result.x, result.fun

        return best_x