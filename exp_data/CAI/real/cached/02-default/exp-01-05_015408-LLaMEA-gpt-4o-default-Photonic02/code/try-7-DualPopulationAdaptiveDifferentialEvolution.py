import numpy as np

class DualPopulationAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.mutation_factor_min = 0.3
        self.crossover_rate_min = 0.5
        self.secondary_population_size = self.population_size // 2

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        primary_population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        secondary_population = np.random.uniform(lb, ub, (self.secondary_population_size, self.dim))
        
        primary_fitness = np.array([func(ind) for ind in primary_population])
        secondary_fitness = np.array([func(ind) for ind in secondary_population])
        
        evaluations = self.population_size + self.secondary_population_size
        best_index = np.argmin(primary_fitness)
        best_position = primary_population[best_index]

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Select three random distinct indices from the primary population
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                x_a, x_b, x_c = primary_population[a], primary_population[b], primary_population[c]

                # Perform mutation
                self.mutation_factor = self.mutation_factor_min + (0.5 * (1 - evaluations / self.budget))
                mutant_vector = x_a + self.mutation_factor * (x_b - x_c)
                mutant_vector = np.clip(mutant_vector, lb, ub)

                # Perform crossover
                self.crossover_rate = self.crossover_rate_min + (0.5 * (1 - evaluations / self.budget))
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                trial_vector = np.where(cross_points, mutant_vector, primary_population[i])

                # Evaluate trial vector
                trial_fitness = func(trial_vector)
                evaluations += 1

                # Selection
                if trial_fitness < primary_fitness[i]:
                    primary_population[i] = trial_vector
                    primary_fitness[i] = trial_fitness

                # Update best position
                if trial_fitness < primary_fitness[best_index]:
                    best_index = i
                    best_position = trial_vector

                if evaluations >= self.budget:
                    break

            # Update secondary population for diversity
            if evaluations < self.budget:
                for j in range(self.secondary_population_size):
                    indices = list(range(self.secondary_population_size))
                    indices.remove(j)
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    y_a, y_b, y_c = secondary_population[a], secondary_population[b], secondary_population[c]

                    mutant_vector = y_a + self.mutation_factor * (y_b - y_c)
                    mutant_vector = np.clip(mutant_vector, lb, ub)

                    cross_points = np.random.rand(self.dim) < self.crossover_rate
                    trial_vector = np.where(cross_points, mutant_vector, secondary_population[j])

                    trial_fitness = func(trial_vector)
                    evaluations += 1

                    if trial_fitness < secondary_fitness[j]:
                        secondary_population[j] = trial_vector
                        secondary_fitness[j] = trial_fitness

                    if trial_fitness < primary_fitness[best_index]:
                        best_index = np.argmin(np.append(primary_fitness, secondary_fitness))
                        best_position = trial_vector if best_index >= self.population_size else primary_population[best_index]

                    if evaluations >= self.budget:
                        break

        return best_position, primary_fitness[best_index]