import numpy as np
import random
from scipy.optimize import differential_evolution

class CSEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.found_solution = None
        self.best_fitness = float('inf')

    def __call__(self, func):
        if self.found_solution is not None:
            return self.found_solution

        # Initialize the population with random points in the search space
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))

        for _ in range(self.budget):
            # Evaluate the fitness of each point in the population
            fitness = [func(point) for point in population]

            # Select the fittest points
            fittest_points = np.array(population[np.argsort(fitness)])

            # Create a new population by adapting the fittest points
            new_population = np.zeros((self.budget, self.dim))
            for i in range(self.budget):
                # Randomly select a parent from the fittest points
                parent = random.choice(fittest_points[:int(self.budget/2)])

                # Randomly select a child from the rest of the population
                child = random.choice(population)

                # Create a new point by averaging the parent and child
                new_point = (parent + child) / 2

                # Add the new point to the new population
                new_population[i] = new_point

            # Update the population
            population = new_population

            # Check if a solution has been found
            if self.check_solution(func, population):
                self.found_solution = population[np.argmin(fitness)]
                self.best_fitness = min(fitness)
                break

        # Refine the solution using mutation
        if self.found_solution is not None:
            for _ in range(int(self.budget * 0.3)):
                # Randomly select a point in the population
                point = random.choice(population)

                # Randomly select a mutation operator (e.g. addition, subtraction, multiplication, division)
                operator = random.choice(['add','sub','mul', 'div'])

                # Apply the mutation operator
                if operator == 'add':
                    new_point = point + np.random.uniform(-1.0, 1.0)
                elif operator =='sub':
                    new_point = point - np.random.uniform(-1.0, 1.0)
                elif operator =='mul':
                    new_point = point * np.random.uniform(0.5, 1.5)
                elif operator == 'div':
                    new_point = point / np.random.uniform(0.5, 1.5)

                # Add the new point to the population
                population = np.vstack((population, new_point))

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