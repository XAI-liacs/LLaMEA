import numpy as np
from scipy.optimize import minimize

class EnhancedHybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        
        sampling_budget = self.budget // 8
        samples = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=(sampling_budget, self.dim))
        
        evaluations = [func(sample) for sample in samples]
        
        decay_schedule = np.exp(-np.linspace(0, 3, sampling_budget))  # Exponential decay schedule
        weighted_evaluations = np.array(evaluations) * decay_schedule
        best_index = np.argmin(weighted_evaluations)
        best_sample = samples[best_index]
        
        perturbation_scale = 0.05 * np.var(evaluations)  # Adaptive scaling based on variance
        gaussian_perturbation = np.random.normal(loc=0.0, scale=perturbation_scale, size=self.dim)
        refined_start = np.clip(best_sample + gaussian_perturbation, func.bounds.lb, func.bounds.ub)

        remaining_budget = self.budget - sampling_budget
        eval_counter = 0

        def callback(xk):
            nonlocal eval_counter
            eval_counter += 1
            if eval_counter >= remaining_budget:
                raise StopIteration

        try:
            result_bfgs = minimize(
                fun=func,
                x0=refined_start,
                method="L-BFGS-B",
                bounds=bounds,
                options={'maxfun': remaining_budget // 2},
                callback=callback
            )
            if eval_counter < remaining_budget:
                result_nelder = minimize(
                    fun=func,
                    x0=result_bfgs.x,
                    method="Nelder-Mead",
                    options={'maxfev': remaining_budget - eval_counter},
                    callback=callback
                )
                result = result_nelder
            else:
                result = result_bfgs
        except StopIteration:
            result = {'x': func(refined_start), 'fun': func(refined_start)}
        
        return result.x