import numpy as np
from scipy.optimize import minimize

class DynamicLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
    
    def __call__(self, func):
        lower_bounds = np.array(func.bounds.lb)
        upper_bounds = np.array(func.bounds.ub)
        best_solution = None
        best_value = float('inf')
        
        num_initial_samples = max(5, self.budget // 3)  # Adjusted initial sampling
        samples = np.random.uniform(lower_bounds, upper_bounds, (num_initial_samples, self.dim))
        for sample in samples:
            if self.evaluations < self.budget:
                value = func(sample)
                self.evaluations += 1
                if value < best_value:
                    best_value = value
                    best_solution = sample
        
        while self.evaluations < self.budget:
            def local_obj(x):
                if self.evaluations >= self.budget:
                    return float('inf')
                value = func(x)
                self.evaluations += 1
                return value
            
            res = minimize(local_obj, best_solution, method='L-BFGS-B', bounds=list(zip(lower_bounds, upper_bounds)))
            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x
            
            epsilon = 0.15 * (upper_bounds - lower_bounds)  # Increased epsilon for broader exploration
            lower_bounds = np.maximum(lower_bounds, best_solution - epsilon)
            upper_bounds = np.minimum(upper_bounds, best_solution + epsilon)
            
            # Introduce restart mechanism
            if self.evaluations < self.budget // 2:
                random_restart = np.random.uniform(lower_bounds, upper_bounds)
                if func(random_restart) < best_value:
                    best_solution = random_restart
                    best_value = func(random_restart)
                    self.evaluations += 1
        
        return best_solution, best_value