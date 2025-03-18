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

        # Enhanced PSO step
        inertia_weight = 0.5 + np.random.rand() * 0.5
        pso_samples = np.random.uniform(func.bounds.lb, func.bounds.ub, (5, self.dim))
        pso_velocities = np.random.uniform(-0.1, 0.1, (5, self.dim))
        pso_evals = [func(sample) for sample in pso_samples]
        remaining_budget -= 5
        pso_best_index = np.argmin(pso_evals)
        pso_best_solution = pso_samples[pso_best_index]

        # Update velocity and position
        for i in range(5):
            pso_velocities[i] = inertia_weight * pso_velocities[i] + 0.5 * (pso_best_solution - pso_samples[i])
            pso_samples[i] += pso_velocities[i]
            pso_evals[i] = func(pso_samples[i])
            remaining_budget -= 1
        
        if min(pso_evals) < best_value:
            best_solution = pso_samples[np.argmin(pso_evals)]
            best_value = min(pso_evals)

        # Iteratively refine solution using local optimizer
        while remaining_budget > 0:
            # Dynamic local optimizer selection
            local_optimizer = 'BFGS' if np.random.rand() > 0.5 else 'nelder-mead'
            options = {'maxiter': min(remaining_budget, 50), 'adaptive': True}

            # Perform local optimization
            result = minimize(
                func, best_solution, method=local_optimizer,
                bounds=list(zip(func.bounds.lb, func.bounds.ub)),
                options=options
            )
            
            # Update remaining budget and best solution found
            remaining_budget -= result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            
            # Update bounds to be closer to the best solution
            func.bounds.lb = np.maximum(func.bounds.lb, best_solution - 0.1 * (func.bounds.ub - func.bounds.lb))
            func.bounds.ub = np.minimum(func.bounds.ub, best_solution + 0.1 * (func.bounds.ub - func.bounds.lb))
            
            # Early stopping if budget is exhausted
            if remaining_budget <= 0:
                break
        
        return best_solution