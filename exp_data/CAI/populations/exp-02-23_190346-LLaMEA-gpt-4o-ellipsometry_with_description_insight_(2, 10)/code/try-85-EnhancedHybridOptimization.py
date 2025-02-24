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

        # Use Sobol sequences for more uniform sampling
        sampling_budget = self.budget // 8
        sobol = qmc.Sobol(d=self.dim, scramble=True)
        samples = qmc.scale(sobol.random_base2(m=int(np.log2(sampling_budget))), func.bounds.lb, func.bounds.ub)

        # Evaluate the function at these samples
        evaluations = [func(sample) for sample in samples]

        # Select the best initial guess with weighted evaluations
        decay_factor = np.linspace(1, 0.5, sampling_budget)
        weighted_evaluations = np.array(evaluations) * decay_factor
        best_index = np.argmin(weighted_evaluations)
        best_sample = samples[best_index]

        # Refined perturbation scaling 
        perturbation_scale = 0.1 * np.log1p(func.bounds.ub - func.bounds.lb) / evaluations[best_index]
        gaussian_perturbation = np.random.normal(loc=0.0, scale=perturbation_scale, size=self.dim)
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