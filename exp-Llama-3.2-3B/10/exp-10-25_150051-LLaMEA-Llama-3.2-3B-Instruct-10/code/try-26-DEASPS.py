import numpy as np
import random

class DEASPS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.adaptive_step_size = 0.5
        self.adaptive_population_size = 0.5
        self.adaptive_mutation_rate = 0.1
        self.adaptive_crossover_rate = 0.1
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

        for i in range(self.budget):
            # Calculate fitness values
            fitness = np.array([func(x) for x in population])

            # Select best individual
            best_individual = population[np.argmin(fitness)]

            # Create new population
            new_population = population + self.adaptive_step_size * np.random.uniform(-1, 1, (self.population_size, self.dim))

            # Adaptive population size
            if np.random.rand() < self.adaptive_population_size:
                new_population = new_population[:int(self.population_size * self.adaptive_population_size)]

            # Adaptive mutation rate
            mutation_mask = np.random.rand(self.population_size) < self.adaptive_mutation_rate
            new_population[mutation_mask] += np.random.uniform(-1, 1, (mutation_mask.sum(), self.dim))

            # Adaptive crossover rate
            crossover_mask = np.random.rand(self.population_size) < self.adaptive_crossover_rate
            new_population[crossover_mask] = np.random.choice(new_population[crossover_mask], size=(crossover_mask.sum(), self.dim), replace=False)

            # Replace worst individual with best individual
            population[np.argsort(fitness)] = np.concatenate((population[np.argsort(fitness)], new_population[np.argsort(fitness)]))

        # Return best individual
        return population[np.argmin(fitness)]

# Example usage:
def func(x):
    return np.sum(x**2)

de = DEASPS(budget=100, dim=10)
best_x = de(func)
print("Best x:", best_x)
print("Best f(x):", func(best_x))