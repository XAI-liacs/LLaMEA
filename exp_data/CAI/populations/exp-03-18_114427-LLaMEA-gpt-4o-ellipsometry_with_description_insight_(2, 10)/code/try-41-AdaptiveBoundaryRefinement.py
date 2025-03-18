import numpy as np
from scipy.optimize import minimize

class AdaptiveBoundaryRefinement:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        remaining_budget = self.budget

        # Uniform sampling for the initial broad search
        random_samples = np.random.uniform(func.bounds.lb, func.bounds.ub, (10, self.dim))
        sample_evals = [func(sample) for sample in random_samples]
        remaining_budget -= 10

        # Choose the best initial guess from the samples
        best_index = np.argmin(sample_evals)
        best_solution = random_samples[best_index]
        best_value = sample_evals[best_index]

        # Enhanced PSO step with adaptive particle count and differential mutation
        pso_count = max(3, int(0.1 * remaining_budget))
        pso_samples = np.random.uniform(func.bounds.lb, func.bounds.ub, (pso_count, self.dim))
        for i in range(pso_count):
            if i % 2 == 0:  # Apply differential mutation to half of the particles
                pso_samples[i] += 0.1 * np.random.randn(self.dim) * (random_samples[i % 10] - random_samples[(i+1) % 10])
        pso_evals = [func(sample) for sample in pso_samples]
        remaining_budget -= pso_count
        pso_best_index = np.argmin(pso_evals)
        pso_best_solution = pso_samples[pso_best_index]

        if pso_evals[pso_best_index] < best_value:
            best_solution = pso_best_solution
            best_value = pso_evals[pso_best_index]

        # Iteratively refine solution using local optimizer
        while remaining_budget > 0:
            # Define a gradient-based local optimization strategy for precision
            local_optimizer = 'BFGS' if remaining_budget > 10 else 'Nelder-Mead'
            options = {'maxiter': min(remaining_budget, 50), 'disp': True}

            # Perform local optimization
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
            
            # Updated bounds adaptation strategy for a tighter focus on optimal region
            adjustment_factor = np.clip(remaining_budget / self.budget, 0.05, 0.2)
            func.bounds.lb = np.maximum(func.bounds.lb, best_solution - adjustment_factor * (func.bounds.ub - func.bounds.lb))
            func.bounds.ub = np.minimum(func.bounds.ub, best_solution + adjustment_factor * (func.bounds.ub - func.bounds.lb))
            
            # Early stopping if budget is exhausted
            if remaining_budget <= 0:
                break
        
        return best_solution