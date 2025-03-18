import numpy as np

class AdaptiveDifferentialRandomWalk:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initial population-based approach
        population_size = 5
        population = [np.random.uniform(lb, ub, self.dim) for _ in range(population_size)]
        fitness_values = [func(individual) for individual in population]
        best_index = np.argmin(fitness_values)
        best_solution = population[best_index]
        best_value = fitness_values[best_index]
        evaluations = population_size
        no_improvement_counter = 0
        
        while evaluations < self.budget:
            # Adaptive exploration factor
            exploration_factor = (0.1 + 0.1 * np.sin(5 * np.pi * evaluations / self.budget)) * (0.5 + 0.5 * (best_value / max(fitness_values))) * np.exp(-evaluations / (0.1 * self.budget))
            # Fitness-based dynamic inertia
            inertia_weight = (0.8 + 0.1 * np.cos(4 * np.pi * evaluations / self.budget)) * (1 - (best_value - min(fitness_values)) / (max(fitness_values) - min(fitness_values) + 1e-9)) * np.exp(-evaluations / (0.05 * self.budget))
            
            for i in range(population_size):
                if np.random.rand() < 0.10:
                    trial_solution = np.random.uniform(lb, ub, self.dim)
                else:
                    improvement_scale = 1 - (no_improvement_counter / (self.budget * 0.05))
                    trial_solution = population[i] + inertia_weight * improvement_scale * np.random.uniform(-1, 1, self.dim) * (ub - lb) * exploration_factor
                trial_solution = np.clip(trial_solution, lb, ub)
                trial_value = func(trial_solution)
                evaluations += 1

                if trial_value < fitness_values[i]:
                    population[i] = trial_solution
                    fitness_values[i] = trial_value
                    if trial_value < best_value:
                        best_solution = trial_solution
                        best_value = trial_value
                        no_improvement_counter = 0
                else:
                    no_improvement_counter += 1
                    if no_improvement_counter > self.budget * 0.06:
                        population[i] = np.random.uniform(lb, ub, self.dim)

        return best_solution