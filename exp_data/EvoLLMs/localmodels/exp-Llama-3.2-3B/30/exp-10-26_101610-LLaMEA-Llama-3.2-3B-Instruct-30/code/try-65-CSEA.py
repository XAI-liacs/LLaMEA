import numpy as np
import random

class CSEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.found_solution = None
        self.best_fitness = float('inf')
        self.drift_rate = 0.3
        self.population_size = 100
        self.mutation_rate = 0.1

    def __call__(self, func):
        if self.found_solution is not None:
            return self.found_solution

        # Initialize the population with random points in the search space
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        for _ in range(self.budget):
            # Evaluate the fitness of each point in the population
            fitness = [func(point) for point in population]

            # Select the fittest points
            fittest_points = np.array(population[np.argsort(fitness)])

            # Create a new population by adapting the fittest points
            new_population = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                # Randomly select a parent from the fittest points
                parent = random.choice(fittest_points[:int(self.population_size/2)])

                # Randomly select a child from the rest of the population
                child = random.choice(population)

                # Create a new point by averaging the parent and child
                new_point = (parent + child) / 2

                # Add the new point to the new population
                new_population[i] = new_point

            # Apply genetic drift
            for i in range(self.population_size):
                if random.random() < self.drift_rate:
                    new_population[i] = np.random.uniform(-5.0, 5.0, self.dim)

            # Apply mutation
            for i in range(self.population_size):
                if random.random() < self.mutation_rate:
                    new_population[i] = np.random.uniform(-5.0, 5.0, self.dim)

            # Update the population
            population = new_population

            # Check if a solution has been found
            if self.check_solution(func, population):
                self.found_solution = population[np.argmin(fitness)]
                self.best_fitness = min(fitness)
                break

        return self.found_solution

    def check_solution(self, func, population):
        # Check if the fitness of the population is within a certain tolerance
        fitness = [func(point) for point in population]
        tolerance = 1e-6
        if np.all(np.abs(np.array(fitness) - self.best_fitness) < tolerance):
            return True
        return False

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2
csea = CSEA(budget, dim)
solution = csea(func)
print(solution)