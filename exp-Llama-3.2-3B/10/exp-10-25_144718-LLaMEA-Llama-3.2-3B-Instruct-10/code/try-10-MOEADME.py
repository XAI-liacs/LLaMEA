import numpy as np
import random

class MOEADME:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        self.fitness_values = []
        self.differential_evolution = 0.5

    def __call__(self, func):
        # Initialize the population with random points
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        # Evaluate the fitness of each point in the population
        for i in range(self.population_size):
            fitness = func(population[i])
            self.fitness_values.append((population[i], fitness))

        # Sort the population based on fitness
        self.fitness_values.sort(key=lambda x: x[1])

        # Select the best points for the next generation
        next_generation = self.fitness_values[:int(self.population_size * 0.2)]

        # Perform crossover and mutation
        for i in range(self.population_size):
            if random.random() < self.crossover_rate:
                parent1, parent2 = random.sample(next_generation, 2)
                child = np.mean([parent1[0], parent2[0]], axis=0)
                if random.random() < self.mutation_rate:
                    child += np.random.uniform(-0.1, 0.1, self.dim)
            else:
                child = next_generation[i][0]

            # Evaluate the fitness of the child
            fitness = func(child)
            self.fitness_values.append((child, fitness))

        # Sort the population based on fitness
        self.fitness_values.sort(key=lambda x: x[1])

        # Differential Evolution
        for i in range(self.population_size):
            if i < self.population_size - self.population_size // 2:
                # Generate the differential vector
                differential_vector = np.zeros(self.dim)
                for j in range(self.dim):
                    differential_vector[j] = np.random.uniform(-1, 1)
                # Update the individual
                individual = next_generation[i]
                for j in range(self.dim):
                    individual[j] += differential_vector[j] * np.random.uniform(0, 1)

            # Evaluate the fitness of the individual
            fitness = func(individual)
            self.fitness_values.append((individual, fitness))

        # Sort the population based on fitness
        self.fitness_values.sort(key=lambda x: x[1])

        # Return the best point in the population
        return self.fitness_values[-1][0]

# Example usage:
def func(x):
    return np.sum(x**2)

meeadme = MOEADME(budget=100, dim=10)
best_point = meeadme(func)
print(best_point)