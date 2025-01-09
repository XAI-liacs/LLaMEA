import numpy as np

class LAMS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight = 0.7
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.mutation_factor_min = 0.05
        self.mutation_factor_max = 0.2
        self.iterations = budget // self.population_size

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        velocities = np.random.rand(self.population_size, self.dim) * 0.1 * (ub - lb)
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size

        personal_best_positions = np.copy(population)
        personal_best_fitness = np.copy(fitness)

        global_best_idx = np.argmin(fitness)
        global_best_position = population[global_best_idx]
        global_best_fitness = fitness[global_best_idx]

        for iteration in range(self.iterations):
            if evals >= self.budget:
                break

            # Update velocities and positions
            r1, r2 = np.random.rand(2)
            velocities = (self.inertia_weight * velocities 
                          + self.cognitive_weight * r1 * (personal_best_positions - population) 
                          + self.social_weight * r2 * (global_best_position - population))
            new_population = np.clip(population + velocities, lb, ub)

            # Evaluate new population
            new_fitness = np.array([func(ind) for ind in new_population])
            evals += self.population_size

            # Update personal bests
            better_mask = new_fitness < personal_best_fitness
            personal_best_positions[better_mask] = new_population[better_mask]
            personal_best_fitness[better_mask] = new_fitness[better_mask]

            # Update global best
            min_new_fitness_idx = np.argmin(new_fitness)
            if new_fitness[min_new_fitness_idx] < global_best_fitness:
                global_best_position = new_population[min_new_fitness_idx]
                global_best_fitness = new_fitness[min_new_fitness_idx]

            # Adaptive mutation factor
            progress_ratio = iteration / self.iterations
            mutation_factor = self.mutation_factor_min + (self.mutation_factor_max - self.mutation_factor_min) * (1 - progress_ratio)

            # Apply mutation strategy
            mutation_prob = np.random.rand(self.population_size, self.dim)
            mutation_mask = mutation_prob < mutation_factor
            mutated_population = np.where(mutation_mask, 
                                          np.random.rand(self.population_size, self.dim) * (ub - lb) + lb, 
                                          new_population)

            # Evaluate mutated individuals
            mutated_fitness = np.array([func(ind) for ind in mutated_population])
            evals += np.sum(mutation_mask)

            # Select between mutated and new individuals
            improved_mask = mutated_fitness < new_fitness
            population[improved_mask] = mutated_population[improved_mask]
            fitness[improved_mask] = mutated_fitness[improved_mask]

            # Re-evaluate global best from possible mutated improvements
            if np.min(fitness) < global_best_fitness:
                global_best_idx = np.argmin(fitness)
                global_best_position = population[global_best_idx]
                global_best_fitness = fitness[global_best_idx]

        return global_best_position, global_best_fitness