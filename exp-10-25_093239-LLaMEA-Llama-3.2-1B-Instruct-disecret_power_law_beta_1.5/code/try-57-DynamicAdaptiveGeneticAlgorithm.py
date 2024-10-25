import random
import math
import numpy as np

class DynamicAdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = [random.uniform(-5.0, 5.0) for _ in range(self.population_size)]
        self.fitnesses = [0] * self.population_size
        self.population_history = []

    def __call__(self, func):
        for _ in range(self.budget):
            # Adaptive sampling: select the next individual based on the fitness and the dimension
            next_individual = self.select_next_individual()

            # Evaluate the function at the next individual
            fitness = func(next_individual)

            # Update the fitness and the population
            self.fitnesses[self.population_size - 1] += fitness
            self.population[self.population_size - 1] = next_individual

            # Ensure the fitness stays within the bounds
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

            # Store the history of the population
            self.population_history.append(self.population)

            # Update the best individual
            best_individual = max(self.population, key=lambda x: self.fitnesses[x])
            best_fitness = self.fitnesses[x]
            if best_fitness < self.fitnesses[self.population_size - 1]:
                self.population[self.population_size - 1] = best_individual
                self.fitnesses[self.population_size - 1] = best_fitness

        # Return the best individual
        return self.population[0]

    def select_next_individual(self):
        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        # and then use adaptive sampling to refine the strategy
        # Select the individual with the highest fitness
        individual = max(self.population, key=lambda x: self.fitnesses[x])
        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        # and then use adaptive sampling to refine the strategy
        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        next_individual = random.choices([individual, best_individual], weights=[0.5, 0.5], k=1)[0]
        return next_individual

# One-line description: "Dynamic Adaptive Genetic Algorithm with Adaptive Sampling"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting.