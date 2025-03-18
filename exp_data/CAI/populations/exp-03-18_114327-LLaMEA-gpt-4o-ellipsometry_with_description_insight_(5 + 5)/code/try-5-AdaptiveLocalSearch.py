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
        num_initial_samples = min(15, self.budget // 3)  # Adjusted number of initial samples
        population = self._initialize_population(bounds, num_initial_samples)
        
        evaluated_points = [(x, self._evaluate(func, x)) for x in population]
        evaluated_points.sort(key=lambda x: x[1])
        best_x, best_val = evaluated_points[0]

        additional_samples = 5  # New search direction
        for i in range(additional_samples):
            direction = np.random.randn(self.dim)
            direction /= np.linalg.norm(direction)
            new_sample = best_x + direction * 0.1 * (bounds.ub - bounds.lb)
            evaluated_points.append((new_sample, self._evaluate(func, new_sample)))
        
        evaluated_points.sort(key=lambda x: x[1])
        best_x, best_val = evaluated_points[0]

        for x, _ in evaluated_points[:3]:
            result = minimize(lambda x: self._evaluate(func, x), x, 
                              method='L-BFGS-B', bounds=[(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)])
            
            if result.fun < best_val:
                best_x, best_val = result.x, result.fun

        return best_x