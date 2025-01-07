import numpy as np

class AdaptiveHybridGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_rate_initial = 0.1
        self.mutation_rate_final = 0.02
        self.crossover_rate_initial = 0.9
        self.crossover_rate_final = 0.6

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness_scores = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        best_index = np.argmin(fitness_scores)
        best_position = population[best_index]

        while evaluations < self.budget:
            # Adaptively change mutation and crossover rates
            progress_ratio = evaluations / self.budget
            mutation_rate = self.mutation_rate_initial * (1 - progress_ratio) + self.mutation_rate_final * progress_ratio
            crossover_rate = self.crossover_rate_initial * (1 - progress_ratio) + self.crossover_rate_final * progress_ratio

            new_population = []

            for _ in range(self.population_size // 2):
                # Select two parents
                parents_indices = np.random.choice(self.population_size, 2, replace=False)
                parent1, parent2 = population[parents_indices]

                # Crossover
                if np.random.rand() < crossover_rate:
                    crossover_point = np.random.randint(1, self.dim)
                    offspring1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                    offspring2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
                else:
                    offspring1, offspring2 = parent1.copy(), parent2.copy()

                # Mutation
                for offspring in [offspring1, offspring2]:
                    if np.random.rand() < mutation_rate:
                        mutation_vector = np.random.normal(0, 0.1, self.dim)
                        offspring += mutation_vector
                        offspring = np.clip(offspring, lb, ub)

                new_population.extend([offspring1, offspring2])

            # Evaluate new population
            new_fitness_scores = np.array([func(ind) for ind in new_population])
            evaluations += self.population_size

            # Combine and select best individuals
            combined_population = np.vstack((population, new_population))
            combined_fitness_scores = np.concatenate((fitness_scores, new_fitness_scores))
            best_indices = combined_fitness_scores.argsort()[:self.population_size]
            population = combined_population[best_indices]
            fitness_scores = combined_fitness_scores[best_indices]

            # Update best position found
            best_index = np.argmin(fitness_scores)
            best_position = population[best_index]

        return best_position, fitness_scores[best_index]