import numpy as np
from scipy.optimize import minimize

class AdaptiveOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        num_initial_samples = min(5, self.budget // 3)
        
        # Step 1: Differential Evolution for Diverse Initial Sampling
        def differential_evolution():
            population = np.random.uniform(
                low=func.bounds.lb, 
                high=func.bounds.ub, 
                size=(num_initial_samples, self.dim)
            )
            f_values = np.array([func(ind) for ind in population])
            for _ in range(10):  # Fixed small number of DE steps
                for i in range(num_initial_samples):
                    idxs = np.random.choice(np.arange(num_initial_samples), 3, replace=False)
                    x1, x2, x3 = population[idxs]
                    scaling_factor = 0.8 + 0.2 * np.random.rand()  # Dynamic scaling factor
                    mutant = np.clip(x1 + scaling_factor * (x2 - x3), func.bounds.lb, func.bounds.ub)
                    if func(mutant) < f_values[i]:  # Stochastic acceptance
                        population[i] = mutant
                        f_values[i] = func(mutant)
            best_idx = np.argmin(f_values)
            return population[best_idx], f_values[best_idx]
        
        best_initial_sample, best_initial_value = differential_evolution()
        remaining_budget = self.budget - num_initial_samples - 10
        
        # Step 2: Local Optimization with L-BFGS-B
        if remaining_budget > 0:
            def local_objective(x):
                return func(x)
            
            result = minimize(
                local_objective, 
                best_initial_sample, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': remaining_budget}
            )
            
            if result.fun < best_initial_value:
                return result.x
        
        return best_initial_sample

# Example usage:
# Assume func is a black-box function with attributes bounds.lb and bounds.ub
# optimizer = AdaptiveOptimizer(budget=100, dim=2)
# best_parameters = optimizer(func)