import numpy as np
from scipy.optimize import minimize

class AdaptiveOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        num_initial_samples = min(5, self.budget // 4) 
        
        # Step 1: Differential Evolution for Diverse Initial Sampling
        def differential_evolution():
            population = np.random.uniform(
                low=func.bounds.lb, 
                high=func.bounds.ub, 
                size=(num_initial_samples, self.dim)
            )
            f_values = np.array([func(ind) for ind in population])
            decay_factor = 0.95
            adaptive_scaling_factor = 0.9
            for gen in range(10):
                for i in range(num_initial_samples):
                    idxs = np.random.choice(np.arange(num_initial_samples), 3, replace=False)
                    x1, x2, x3 = population[idxs]
                    scaling_factor = (adaptive_scaling_factor + 0.1 * np.random.rand()) * (decay_factor ** gen)
                    mutant = np.clip(x1 + scaling_factor * (x2 - x3), func.bounds.lb, func.bounds.ub)
                    if func(mutant) < f_values[i]: 
                        population[i] = mutant
                        f_values[i] = func(mutant)
            best_idx = np.argmin(f_values)
            return population[best_idx], f_values[best_idx]
        
        def particle_swarm_optimization():
            num_particles = 5
            velocity = np.zeros_like(population)
            personal_best = population.copy()
            personal_best_value = f_values.copy()
            global_best = population[np.argmin(f_values)].copy()
            global_best_value = np.min(f_values)

            for _ in range(10):
                r1, r2 = np.random.rand(2)
                velocity = 0.5 * velocity + r1 * (personal_best - population) + r2 * (global_best - population)
                population += velocity
                population = np.clip(population, func.bounds.lb, func.bounds.ub)
                current_value = np.array([func(ind) for ind in population])
                
                better_mask = current_value < personal_best_value
                personal_best[better_mask] = population[better_mask]
                personal_best_value[better_mask] = current_value[better_mask]
                
                if np.min(current_value) < global_best_value:
                    global_best = population[np.argmin(current_value)]
                    global_best_value = np.min(current_value)

            return global_best, global_best_value
        
        best_initial_sample, best_initial_value = differential_evolution()
        remaining_budget = self.budget - num_initial_samples - 10
        
        if remaining_budget > 0:
            def local_objective(x):
                return func(x)
            
            result = minimize(
                local_objective, 
                best_initial_sample, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': remaining_budget, 'ftol': 1e-9}
            )
            
            if result.fun < best_initial_value:
                return result.x
        
        return best_initial_sample

# Example usage:
# Assume func is a black-box function with attributes bounds.lb and bounds.ub
# optimizer = AdaptiveOptimizer(budget=100, dim=2)
# best_parameters = optimizer(func)