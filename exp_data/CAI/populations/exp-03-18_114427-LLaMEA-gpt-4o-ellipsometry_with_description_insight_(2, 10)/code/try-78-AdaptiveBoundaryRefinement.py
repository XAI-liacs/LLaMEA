import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class AdaptiveBoundaryRefinement:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        remaining_budget = self.budget

        # Change 1: Use Latin Hypercube Sampling with safe bounds
        sampler = qmc.LatinHypercube(d=self.dim)
        random_samples = qmc.scale(sampler.random(10), func.bounds.lb, func.bounds.ub)
        sample_evals = [func(sample) for sample in random_samples]
        remaining_budget -= 10

        # Choose the best initial guess from the samples
        best_index = np.argmin(sample_evals)
        best_solution = random_samples[best_index]
        best_value = sample_evals[best_index]

        # Enhanced PSO step with adaptive particle count
        pso_count = max(3, int(0.1 * remaining_budget))
        pso_samples = np.random.uniform(func.bounds.lb, func.bounds.ub, (pso_count, self.dim))
        pso_evals = [func(sample) for sample in pso_samples]
        remaining_budget -= pso_count
        pso_best_index = np.argmin(pso_evals)
        pso_best_solution = pso_samples[pso_best_index]

        if pso_evals[pso_best_index] < best_value:
            best_solution = pso_best_solution
            best_value = pso_evals[pso_best_index]

        # Iteratively refine solution using local optimizer
        while remaining_budget > 0:
            # Change 2: Adjust local optimizer choice logic
            local_optimizer = 'BFGS' if remaining_budget > 5 else 'Nelder-Mead'
            options = {'maxiter': min(remaining_budget, 50), 'disp': False, 'learning_rate': 0.1}  # Change 3: Set disp=False
            result = minimize(
                func, best_solution, method=local_optimizer,
                bounds=list(zip(func.bounds.lb, func.bounds.ub)) if local_optimizer != 'BFGS' else None,
                options=options
            )
            
            # Update remaining budget and best solution found
            remaining_budget -= result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            
            # Change 4: Ensure bounds adjustment is safe
            adjustment_factor = 0.05 * (func.bounds.ub - func.bounds.lb)
            new_lb = np.maximum(func.bounds.lb, best_solution - adjustment_factor)
            new_ub = np.minimum(func.bounds.ub, best_solution + adjustment_factor)
            func.bounds.lb = np.minimum(func.bounds.ub - adjustment_factor, new_lb)
            func.bounds.ub = np.maximum(func.bounds.lb + adjustment_factor, new_ub)
            
            # Early stopping if budget is exhausted
            if remaining_budget <= 0:
                break
        
        return best_solution