import numpy as np

class EnhancedImprovedDynamicSearchSpaceExploration_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lower_bound = -5.0
        upper_bound = 5.0
        best_solution = np.random.uniform(lower_bound, upper_bound, size=self.dim)
        best_fitness = func(best_solution)
        step_size = 0.1 * (upper_bound - lower_bound)  # Adaptive step size
        for _ in range(self.budget):
            # Introduce Levy flights for exploring new solutions with improved step size adaptation
            levy_step = np.random.standard_cauchy(size=self.dim) / (1.0 + np.abs(np.random.normal(size=self.dim)))
            new_solution = best_solution + levy_step * step_size
            new_solution = np.clip(new_solution, lower_bound, upper_bound)
            new_fitness = func(new_solution)
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
                step_size *= 0.95  # Self-adaptive strategy enhancement
                
            # Differential Evolution strategy with enhanced parameters for improved exploitation
            F = 0.7  # Adjusted Differential weight
            CR = 0.8  # Adjusted Crossover probability
            mutant = best_solution + F * (best_solution - new_solution)
            trial_solution = np.where(np.random.uniform(0, 1, self.dim) < CR, mutant, new_solution)
            trial_fitness = func(trial_solution)
            if trial_fitness < best_fitness:
                best_solution = trial_solution
                best_fitness = trial_fitness
                step_size *= 0.95  # Self-adaptive strategy enhancement
        
        return best_solution