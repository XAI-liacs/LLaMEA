import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class HybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Define the bounds for the optimization
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        
        # Use a portion of the budget for uniform sampling to get initial guesses
        sampling_budget = self.budget // 10
        sampler = Sobol(d=self.dim, scramble=True)
        samples = sampler.random_base2(m=int(np.log2(sampling_budget)))
        samples = samples * (func.bounds.ub - func.bounds.lb) + func.bounds.lb
        
        # Evaluate the function at these samples
        evaluations = [func(sample) for sample in samples]
        
        # Select the best initial guess
        best_index = np.argmin(evaluations)
        best_sample = samples[best_index]
        
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
                fun=func,  # Change here
                x0=best_sample,
                method="L-BFGS-B",
                bounds=bounds,
                options={'maxfun': remaining_budget},
                callback=callback
            )
        except StopIteration:
            result = {'x': func(best_sample), 'fun': func(best_sample)}
        
        return result.x