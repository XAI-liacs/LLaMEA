import numpy as np

class EnhancedAdaptiveOppositionEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.alpha = 0.5  # Initial opposition learning rate
        self.alpha_decay = 0.995  # Alpha decay factor
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            # Dynamic opposition-based learning with decay
            opposite_population = lb + ub - population + np.random.normal(0, 0.2, population.shape)
            opposite_fitness = np.array([func(ind) for ind in opposite_population])
            evaluations += self.population_size

            # Adaptive parameter tuning
            self.crossover_rate = 0.5 + 0.5 * np.random.rand()
            self.mutation_rate = 0.1 + 0.4 * np.random.rand()

            # Generate offspring
            offspring = []
            for i in range(self.population_size):
                if np.random.rand() < self.crossover_rate:
                    parents = np.random.choice(self.population_size, 2, replace=False)
                    parent1, parent2 = population[parents]
                    child = np.clip(parent1 + self.alpha * (parent2 - parent1), lb, ub)
                    if np.random.rand() < self.mutation_rate:
                        mutation = np.random.uniform(-1, 1, self.dim) * (ub - lb) * 0.05
                        child = np.clip(child + mutation, lb, ub)
                    offspring.append(child)

            offspring = np.array(offspring)
            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)

            # Combine and select next generation with diversity-based elitism
            combined_population = np.vstack((population, opposite_population, offspring))
            combined_fitness = np.hstack((fitness, opposite_fitness, offspring_fitness))
            diversity_indices = np.argsort(combined_fitness)[:self.population_size]
            population = combined_population[diversity_indices]
            fitness = combined_fitness[diversity_indices]
            self.alpha *= self.alpha_decay  # Update alpha

        return population[np.argmin(fitness)]