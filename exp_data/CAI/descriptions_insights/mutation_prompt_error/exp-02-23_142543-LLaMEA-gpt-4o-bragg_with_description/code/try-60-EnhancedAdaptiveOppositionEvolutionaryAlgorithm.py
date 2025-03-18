import numpy as np

class EnhancedAdaptiveOppositionEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.alpha = 0.5
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.differential_weight = 0.8  # Differential Evolution weight

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            # Dynamic opposition-based learning with added perturbation
            opposite_population = lb + ub - population + np.random.normal(0, 0.3, population.shape)
            opposite_fitness = np.array([func(ind) for ind in opposite_population])
            evaluations += self.population_size

            # Adaptive parameter tuning with exploration-focused strategy
            self.crossover_rate = 0.6 + 0.4 * np.random.rand()
            self.mutation_rate = 0.2 + 0.3 * np.random.rand()

            # Generate offspring using Differential Evolution crossover
            offspring = []
            for i in range(self.population_size):
                if np.random.rand() < self.crossover_rate:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    a, b, c = population[indices]
                    mutant_vector = np.clip(a + self.differential_weight * (b - c), lb, ub)
                    if np.random.rand() < self.mutation_rate:
                        mutation = np.random.uniform(-1, 1, self.dim) * (ub - lb) * 0.05
                        mutant_vector = np.clip(mutant_vector + mutation, lb, ub)
                    offspring.append(mutant_vector)

            offspring = np.array(offspring)
            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)

            # Combine and select next generation with elitism
            combined_population = np.vstack((population, opposite_population, offspring))
            combined_fitness = np.hstack((fitness, opposite_fitness, offspring_fitness))
            best_indices = np.argsort(combined_fitness)[:self.population_size]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]

        return population[np.argmin(fitness)]