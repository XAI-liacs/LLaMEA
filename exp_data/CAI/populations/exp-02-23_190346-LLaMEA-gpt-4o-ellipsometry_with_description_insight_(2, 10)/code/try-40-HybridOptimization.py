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
        samples = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=(sampling_budget, self.dim))
        
        # Evaluate the function at these samples
        evaluations = [func(sample) for sample in samples]
        
        # Select the best initial guess
        best_index = np.argmin(evaluations)
        best_sample = samples[best_index]
        
        # Adjust bounds based on the best sample
        bounds = [(max(lb, bs - 0.15*(ub-lb)), min(ub, bs + 0.15*(ub-lb))) for (lb, ub), bs in zip(bounds, best_sample)]  # Slightly increased adjustment range

        # Remaining budget for local optimization
        remaining_budget = self.budget - sampling_budget
        eval_counter = 0

        # Define the callback to limit function evaluations
        def callback(xk):
            nonlocal eval_counter
            eval_counter += 1
            if eval_counter >= remaining_budget:
                raise StopIteration
        
        # Start point refinement using an approximation of the gradient
        gradient_estimate = np.array([(func(best_sample + np.eye(1, self.dim, i).flatten() * 1e-8) - func(best_sample)) / 1e-8 for i in range(self.dim)])  # Improved accuracy of gradient estimation
        refined_start = best_sample - 0.01 * gradient_estimate

        # Use BFGS with dynamically constrained bounds
        try:
            result = minimize(
                fun=func,
                x0=refined_start,  # Changed from 'best_sample' to 'refined_start'
                method="L-BFGS-B",
                bounds=bounds,
                options={'maxfun': remaining_budget},
                callback=callback
            )
        except StopIteration:
            result = {'x': func(refined_start), 'fun': func(refined_start)}  # Changed from 'best_sample' to 'refined_start'}
        
        return result.x