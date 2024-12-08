import numpy as np
import random

class CDEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.crowd_size = 10
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.crowd = np.random.uniform(-5.0, 5.0, (self.crowd_size, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the population
            population_values = np.array([func(x) for x in self.population])

            # Evaluate the crowd
            crowd_values = np.array([func(x) for x in self.crowd])

            # Select the best individuals
            best_indices = np.argsort(population_values)[:, -self.crowd_size:]
            best_crowd_values = crowd_values[best_indices]

            # Select the worst individuals
            worst_indices = np.argsort(population_values)[:, :self.crowd_size]
            worst_population_values = population_values[worst_indices]

            # Update the population
            new_population = []
            for i in range(self.population_size):
                # Select a parent from the best crowd
                parent = random.choice(best_crowd_values)

                # Select a parent from the worst population with 50% probability
                if random.random() < 0.5:
                    parent = random.choice(worst_population_values)

                # Create a child by concatenating the parent with a random individual from the population
                child = np.concatenate((parent, self.population[np.random.choice(self.population_size)]))

                new_population.append(child)

            self.population = np.array(new_population)

            # Update the crowd
            self.crowd = self.population[:self.crowd_size]

            # Perform crossover and mutation
            self.population = self.crossover(self.population)
            self.population = self.mutate(self.population)

    def crossover(self, population):
        # Perform single-point crossover
        offspring = []
        for _ in range(len(population)):
            parent1, parent2 = random.sample(population, 2)
            child = np.concatenate((parent1, parent2[1:]))
            offspring.append(child)
        return np.array(offspring)

    def mutate(self, population):
        # Perform Gaussian mutation
        mutated_population = population + np.random.normal(0, 1, population.shape)
        return np.clip(mutated_population, -5.0, 5.0)