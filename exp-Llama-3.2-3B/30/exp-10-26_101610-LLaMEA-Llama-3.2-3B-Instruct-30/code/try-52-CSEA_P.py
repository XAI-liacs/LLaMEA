import numpy as np
import random

class CSEA_P:
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

                # Add the new point to the new population with a probability of 0.3
                if random.random() < 0.3:
                    new_population[i] = new_point
                else:
                    # Otherwise, randomly select a mutation operator from the list
                    mutation_operators = [self.mut_add, self.mut_sub, self.mut_mul, self.mut_div]
                    new_population[i] = mutation_operators[random.randint(0, len(mutation_operators) - 1)](child)

            # Update the population
            population = new_population

            # Check if a solution has been found
            if self.check_solution(func, population):
                self.found_solution = population[np.argmin(fitness)]
                self.best_fitness = min(fitness)
                break

        return self.found_solution

    def mut_add(self, point):
        # Add a random value between -1 and 1 to the point
        return point + np.random.uniform(-1, 1, self.dim)

    def mut_sub(self, point):
        # Subtract a random value between -1 and 1 from the point
        return point - np.random.uniform(-1, 1, self.dim)

    def mut_mul(self, point):
        # Multiply the point by a random value between 0.5 and 1.5
        return point * np.random.uniform(0.5, 1.5, self.dim)

    def mut_div(self, point):
        # Divide the point by a random value between 0.5 and 1.5
        return point / np.random.uniform(0.5, 1.5, self.dim)

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2
csea_p = CSEA_P(budget, dim)
solution = csea_p(func)
print(solution)