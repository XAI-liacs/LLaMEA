import numpy as np
from scipy.optimize import minimize

class EnhancedHybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Define the bounds for the optimization
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        
        # Use a portion of the budget for adaptive sampling with decaying learning rate
        sampling_budget = self.budget // 8
        decay_learning_rate = np.linspace(1.0, 0.1, sampling_budget)  # Decaying learning rate
        samples = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=(sampling_budget, self.dim))
        
        # Evaluate the function at these samples
        evaluations = [func(sample) for sample in samples * decay_learning_rate[:, None]]
        
        # Select the best initial guess with weighted evaluations
        weighted_evaluations = np.array(evaluations)
        best_index = np.argmin(weighted_evaluations)
        best_sample = samples[best_index]
        
        # Adjust bounds based on the best sample using a dynamically scaled perturbation
        perturbation_scale = 0.1 * np.log1p(1 + (func.bounds.ub - func.bounds.lb)) / (1 + evaluations[best_index])
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

        # Use a hybrid local optimization approach with L-BFGS-B and Nelder-Mead
        try:
            result = minimize(
                fun=func,
                x0=refined_start,
                method="L-BFGS-B",
                bounds=bounds,
                options={'maxfun': remaining_budget // 2},
                callback=callback
            )
            
            # Use Nelder-Mead for further refinement if budget allows
            if eval_counter < remaining_budget:
                result = minimize(
                    fun=func,
                    x0=result.x,
                    method="Nelder-Mead",
                    options={'maxfev': remaining_budget - eval_counter},
                    callback=callback
                )
        except StopIteration:
            result = {'x': func(refined_start), 'fun': func(refined_start)}
        
        return result.x