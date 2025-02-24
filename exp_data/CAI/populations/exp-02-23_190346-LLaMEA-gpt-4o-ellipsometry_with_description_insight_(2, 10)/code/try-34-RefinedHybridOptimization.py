import numpy as np
from scipy.optimize import minimize

class RefinedHybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Define the initial bounds for the optimization
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        
        # Use a portion of the budget for adaptive sampling
        sampling_budget = max(5, self.budget // 6)  # Changed from budget // 8 to budget // 6
        samples = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=(sampling_budget, self.dim))
        
        # Evaluate the function at these samples
        evaluations = [func(sample) for sample in samples]
        
        # Select the best initial guess
        best_index = np.argmin(evaluations)
        best_sample = samples[best_index]

        # Use the variance of the samples to adjust bounds
        sample_std = np.std(samples, axis=0)
        dynamic_bounds = [(max(lb, bs - 0.8*std), min(ub, bs + 0.8*std)) for (lb, ub), bs, std in zip(bounds, best_sample, sample_std)]
        
        # Remaining budget for local optimization
        remaining_budget = self.budget - sampling_budget
        eval_counter = 0

        # Define the callback to limit function evaluations
        def callback(xk):
            nonlocal eval_counter
            eval_counter += 1
            if eval_counter >= remaining_budget:
                raise StopIteration
        
        # Use L-BFGS-B with dynamically constrained bounds
        try:
            result = minimize(
                fun=func,
                x0=best_sample,
                method="L-BFGS-B",
                bounds=dynamic_bounds,
                options={'maxfun': remaining_budget},
                callback=callback
            )
        except StopIteration:
            result = {'x': func(best_sample), 'fun': func(best_sample)}
        
        return result.x