import numpy as np
from scipy.optimize import minimize

class HybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Define the bounds for the optimization
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        
        # Use a portion of the budget for adaptive sampling to get initial guesses
        sampling_budget = self.budget // 8  # Reduced sampling budget
        # Modified sampling to use normal distribution centered at mid-point of bounds
        samples = np.random.normal(loc=(func.bounds.lb + func.bounds.ub) / 2, scale=(func.bounds.ub - func.bounds.lb) / 6, size=(sampling_budget, self.dim))
        
        # Evaluate the function at these samples
        evaluations = [func(sample) for sample in samples]
        
        # Select the best initial guess
        best_index = np.argmin(evaluations)
        best_sample = samples[best_index]
        
        # Adjust bounds based on the best sample
        bounds = [(max(lb, bs - 0.1*(ub-lb)), min(ub, bs + 0.1*(ub-lb))) for (lb, ub), bs in zip(bounds, best_sample)]

        # Remaining budget for local optimization
        remaining_budget = self.budget - sampling_budget
        eval_counter = 0

        # Define the callback to limit function evaluations
        def callback(xk):
            nonlocal eval_counter
            eval_counter += 1
            if eval_counter >= remaining_budget:
                raise StopIteration
        
        # Use BFGS with dynamically constrained bounds
        try:
            result = minimize(
                fun=func,  # Corrected argument from 'func' to 'fun'
                x0=best_sample,
                method="L-BFGS-B",
                bounds=bounds,
                options={'maxfun': remaining_budget},
                callback=callback
            )
        except StopIteration:
            result = {'x': func(best_sample), 'fun': func(best_sample)}
        
        return result.x