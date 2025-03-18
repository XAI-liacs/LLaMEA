import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class APSR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        best_solution = None
        best_value = float('inf')
        remaining_budget = self.budget
        
        num_initial_samples = min(max(7, 4 * self.dim), remaining_budget // 3)
        sampler = Sobol(self.dim, scramble=True)
        samples = lb + (ub - lb) * sampler.random(num_initial_samples)
        remaining_budget -= num_initial_samples
        
        for sample in samples:
            value = func(sample)
            remaining_budget -= 1
            if value < best_value:
                best_value = value
                best_solution = sample
        
        while remaining_budget > 0:
            result = minimize(func, best_solution, method='L-BFGS-B', bounds=zip(lb, ub), options={'maxfun': remaining_budget})
            remaining_budget -= result.nfev
            
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            
            bounds_range = (ub - lb) / 5
            lb = np.maximum(lb, best_solution - bounds_range)
            ub = np.minimum(ub, best_solution + bounds_range)

            # Introduce an adaptive restart mechanism
            if remaining_budget > 0 and np.random.rand() < 0.1:  # Added restart with 10% probability
                restart_sample = lb + (ub - lb) * np.random.rand(self.dim)  # New random sample within current bounds
                value = func(restart_sample)
                remaining_budget -= 1
                if value < best_value:
                    best_value = value
                    best_solution = restart_sample
        
        return best_solution