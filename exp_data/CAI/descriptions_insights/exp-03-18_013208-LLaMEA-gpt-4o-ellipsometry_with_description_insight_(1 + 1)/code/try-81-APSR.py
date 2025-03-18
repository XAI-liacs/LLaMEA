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
        
        # Adjusted initial Sobol sequence sampling
        num_initial_samples = min(max(7, 4 * self.dim), remaining_budget // 3)  # Changed from 5 to 7 for better exploration
        sampler = Sobol(self.dim, scramble=True)
        samples = lb + (ub - lb) * sampler.random(num_initial_samples)
        remaining_budget -= num_initial_samples
        
        # Evaluate initial samples
        for sample in samples:
            value = func(sample)
            remaining_budget -= 1
            if value < best_value:
                best_value = value
                best_solution = sample
        
        # Adaptive Parameter Space Reduction with local optimization
        while remaining_budget > 0:
            # Run local optimization
            result = minimize(func, best_solution, method='L-BFGS-B', bounds=zip(lb, ub), options={'maxfun': remaining_budget})
            remaining_budget -= result.nfev
            
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            
            # Updated bounds adaptation
            bounds_range = (ub - lb) / 5  # Changed from /4 to /5 for more aggressive adaptation
            lb = np.maximum(lb, best_solution - bounds_range)
            ub = np.minimum(ub, best_solution + bounds_range)
        
            # Occasional gradient-based refinement
            if remaining_budget > 0:  # Refine using gradient information
                grad_solution = minimize(func, best_solution, jac=True, method='L-BFGS-B', bounds=zip(lb, ub), options={'maxfun': 1})
                if grad_solution.fun < best_value:
                    best_value = grad_solution.fun
                    best_solution = grad_solution.x
                    remaining_budget -= grad_solution.nfev
        
        return best_solution