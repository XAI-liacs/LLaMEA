import numpy as np
from scipy.optimize import minimize

class EnhancedHybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Define the bounds for the optimization
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        
        # Use a portion of the budget for initial exploratory sampling
        sampling_budget = self.budget // 3  # Adjusted the sampling budget allocation
        samples = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=(sampling_budget, self.dim))
        
        # Evaluate the function at these sample points
        evaluations = [func(sample) for sample in samples]
        
        # Select the top few initial guesses
        sorted_indices = np.argsort(evaluations)
        top_guesses = [samples[i] for i in sorted_indices[:3]]
        
        # Introduce a dynamic perturbation scale based on initial findings
        perturbation_scale = 0.1 * (func.bounds.ub - func.bounds.lb)  # Modified perturbation scale
        refined_samples = [
            np.clip(sample + np.random.normal(loc=0.0, scale=perturbation_scale, size=self.dim), func.bounds.lb, func.bounds.ub)
            for sample in top_guesses
        ]

        # Remaining budget for local optimization
        remaining_budget = self.budget - sampling_budget
        eval_counter = 0

        # Define the callback to limit function evaluations
        def callback(xk):
            nonlocal eval_counter
            eval_counter += 1
            if eval_counter >= remaining_budget:
                raise StopIteration

        # Use BFGS with dynamically constrained bounds for each refined sample
        best_result = None
        best_value = float('inf')
        try:
            for refined_start in refined_samples:
                result = minimize(
                    fun=func,
                    x0=refined_start,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={'maxfun': remaining_budget // len(refined_samples)},
                    callback=callback
                )
                if result.fun < best_value:
                    best_value = result.fun
                    best_result = result
        except StopIteration:
            best_result = {'x': refined_samples[0], 'fun': func(refined_samples[0])}
        
        return best_result.x