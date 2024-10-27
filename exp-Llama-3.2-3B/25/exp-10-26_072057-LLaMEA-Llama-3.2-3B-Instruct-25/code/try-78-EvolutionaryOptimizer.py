import numpy as np
import random
import copy

class EvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        self.swarm_size = 10
        self.adaptive_mutation_rate = 0.25

    def __call__(self, func):
        # Initialize population with random solutions
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        for _ in range(self.budget):
            # Evaluate population
            fitness = np.array([func(x) for x in population])

            # Select fittest individuals
            fittest = population[np.argsort(fitness)]

            # Generate new offspring
            offspring = []
            for _ in range(self.population_size):
                # Select parents
                parent1, parent2 = random.sample(fittest, 2)

                # Crossover
                child = np.concatenate((parent1[:self.dim//2], parent2[self.dim//2:]))

                # Mutate
                if random.random() < self.adaptive_mutation_rate:
                    child += np.random.uniform(-0.1, 0.1, self.dim)

                offspring.append(child)

            # Replace least fit individuals with new offspring
            population = np.array(offspring)

        # Select the best solution from the final population
        best_solution = population[np.argmin(fitness)]

        # Refine the best solution by changing individual lines with a probability of 0.25
        refined_solution = copy.deepcopy(best_solution)
        for i in range(self.dim):
            if random.random() < self.adaptive_mutation_rate:
                refined_solution[i] += np.random.uniform(-0.1, 0.1)

        return refined_solution

# Example usage
def func(x):
    return np.sum(x**2)

optimizer = EvolutionaryOptimizer(budget=100, dim=5)
best_solution = optimizer(func)
print("Best solution:", best_solution)