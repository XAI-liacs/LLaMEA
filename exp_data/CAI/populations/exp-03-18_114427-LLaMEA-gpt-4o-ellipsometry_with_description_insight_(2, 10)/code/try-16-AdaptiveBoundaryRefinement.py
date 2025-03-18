import numpy as np
from scipy.optimize import minimize

class AdaptiveBoundaryRefinement:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        remaining_budget = self.budget
        
        # Uniform sampling for the initial broad search
        adaptive_scale = 0.2 * (func.bounds.ub - func.bounds.lb)  # Introduce adaptive sampling scale
        random_samples = np.random.uniform(func.bounds.lb - adaptive_scale, func.bounds.ub + adaptive_scale, (10, self.dim)) 
        sample_evals = [func(sample) for sample in random_samples]
        remaining_budget -= 10
        
        # Choose the best initial guess from the samples
        best_index = np.argmin(sample_evals)
        best_solution = random_samples[best_index]
        best_value = sample_evals[best_index]
        
        # Iteratively refine solution using local optimizer
        while remaining_budget > 0:
            # Define a local optimization strategy
            local_optimizer = 'nelder-mead'
            options = {'maxiter': min(remaining_budget, 50), 'adaptive': True}

            # Perform local optimization
            result = minimize(
                func, best_solution, method=local_optimizer,
                bounds=list(zip(func.bounds.lb, func.bounds.ub)),
                options=options
            )
            
            # Update remaining budget and best solution found
            remaining_budget -= result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            
            # Update bounds to be closer to the best solution
            func.bounds.lb = np.maximum(func.bounds.lb, best_solution - 0.1 * (func.bounds.ub - func.bounds.lb))
            func.bounds.ub = np.minimum(func.bounds.ub, best_solution + 0.1 * (func.bounds.ub - func.bounds.lb))
            
            # Early stopping if budget is exhausted
            if remaining_budget <= 0:
                break
        
        return best_solution