import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class EnhancedHybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Define the bounds for the optimization
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        
        # Use a portion of the budget for probabilistic adaptive sampling
        sampling_budget = self.budget // 8  
        sampler = qmc.LatinHypercube(d=self.dim)  # Changed to Latin Hypercube sampling
        samples = qmc.scale(sampler.random(n=sampling_budget), func.bounds.lb, func.bounds.ub)
        
        # Evaluate the function at these samples
        evaluations = [func(sample) for sample in samples]
        
        # Select the best initial guess
        best_index = np.argmin(evaluations)
        best_sample = samples[best_index]
        
        # Adjust bounds based on the best sample using a Gaussian perturbation
        perturbation_scale = 0.05 * np.log1p(func.bounds.ub - func.bounds.lb)  
        adaptive_scale = 0.7 + 0.3 * (1 - evaluations[best_index] / np.max(evaluations))  # Adjusted adaptive step size
        gaussian_perturbation = np.random.normal(loc=0.0, scale=perturbation_scale * adaptive_scale, size=self.dim)
        refined_start = np.clip(best_sample + gaussian_perturbation, func.bounds.lb, func.bounds.ub)

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
                fun=func,
                x0=refined_start,
                method="L-BFGS-B",
                bounds=bounds,
                options={'maxfun': remaining_budget},
                callback=callback
            )
        except StopIteration:
            result = {'x': func(refined_start), 'fun': func(refined_start)}
        
        return result.x