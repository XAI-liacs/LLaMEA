import numpy as np
from scipy.optimize import minimize

class AdaptiveOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        num_initial_samples = min(5, self.budget // 3)
        
        # Step 1: Differential Evolution with Adaptive Crossover
        def differential_evolution():
            population = np.random.uniform(
                low=func.bounds.lb, 
                high=func.bounds.ub, 
                size=(num_initial_samples, self.dim)
            )
            f_values = np.array([func(ind) for ind in population])
            for _ in range(15):  # Increased number of DE steps
                for i in range(num_initial_samples):
                    idxs = np.random.choice(np.arange(num_initial_samples), 3, replace=False)
                    x1, x2, x3 = population[idxs]
                    F = np.random.uniform(0.5, 1.0)  # Adaptive scaling factor
                    mutant = np.clip(x1 + F * (x2 - x3), func.bounds.lb, func.bounds.ub)
                    if np.random.rand() < 0.7 or func(mutant) < f_values[i]:  # Adjusted stochastic acceptance
                        population[i] = mutant
                        f_values[i] = func(mutant)
            best_idx = np.argmin(f_values)
            return population[best_idx], f_values[best_idx]
        
        best_initial_sample, best_initial_value = differential_evolution()
        remaining_budget = self.budget - num_initial_samples - 15
        
        # Step 2: Dual-Phase Local Optimization with L-BFGS-B
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