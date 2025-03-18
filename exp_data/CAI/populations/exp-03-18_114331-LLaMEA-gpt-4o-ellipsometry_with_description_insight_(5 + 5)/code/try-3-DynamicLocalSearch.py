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
        
        # Initial uniform sampling
        num_initial_samples = min(10, self.budget // 2)
        samples = np.random.uniform(lower_bounds, upper_bounds, (num_initial_samples, self.dim))
        for sample in samples:
            if self.evaluations < self.budget:
                value = func(sample)
                self.evaluations += 1
                if value < best_value:
                    best_value = value
                    best_solution = sample
                    
        # Refine using local optimization (BFGS)
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
            
            # Dynamically adjust bounds to focus search
            epsilon = 0.1 * (upper_bounds - lower_bounds)
            lower_bounds = np.maximum(lower_bounds, best_solution - epsilon)
            upper_bounds = np.minimum(upper_bounds, best_solution + epsilon)
        
        return best_solution, best_value