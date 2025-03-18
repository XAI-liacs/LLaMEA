import numpy as np

class AdaptiveOppositionEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.alpha = 0.5  # Opposition learning rate
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        opposite_population = lb + ub - population + np.random.uniform(-0.1, 0.1, (self.population_size, self.dim))
        opposite_fitness = np.array([func(ind) for ind in opposite_population])
        evaluations = self.population_size * 2

        while evaluations < self.budget:
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
                        mutation_scale = (ub - lb) * 0.05 * (1 - evaluations / self.budget)
                        mutation = np.random.uniform(-1, 1, self.dim) * mutation_scale
                        child = np.clip(child + mutation, lb, ub)
                    offspring.append(child)

            offspring = np.array(offspring)
            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)

            # Combine and select next generation
            combined_population = np.vstack((population, opposite_population, offspring))
            combined_fitness = np.hstack((fitness, opposite_fitness, offspring_fitness))
            best_indices = np.argsort(combined_fitness)[:self.population_size]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]

            # Update opposite population
            opposite_population = lb + ub - population + np.random.uniform(-0.1, 0.1, (self.population_size, self.dim))
            opposite_fitness = np.array([func(ind) for ind in opposite_population])
            evaluations += self.population_size

        return population[np.argmin(fitness)]