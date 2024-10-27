import numpy as np
import random

class CDEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.f_best = None
        self.x_best = None
        self.f_history = []
        self.x_history = []
        self.change_probability = 0.05

    def __call__(self, func):
        if self.f_best is None:
            self.f_best = func(0)
            self.x_best = 0
            self.f_history.append(self.f_best)
            self.x_history.append(self.x_best)

        # Initialize the population with random points
        population = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))

        # Evaluate the population
        fitness = [func(x) for x in population]

        # Update the best solution
        idx_best = np.argmin(fitness)
        if fitness[idx_best] < self.f_best:
            self.f_best = fitness[idx_best]
            self.x_best = population[idx_best]

        # Perform selection, crossover, and mutation
        for _ in range(self.budget - 1):
            # Selection
            idx = np.random.choice(self.budget, size=2, replace=False)
            parent1, parent2 = population[idx]

            # Crossover
            child = (parent1 + parent2) / 2

            # Mutation
            if random.random() < self.change_probability:
                # Randomly change individual lines with 0.05 probability
                for i in range(self.dim):
                    if random.random() < self.change_probability:
                        child[i] += np.random.uniform(-1.0, 1.0)
                        population[idx[0]][i] += np.random.uniform(-1.0, 1.0)
                        population[idx[1]][i] += np.random.uniform(-1.0, 1.0)

            # Evaluate the child
            fitness_child = func(child)

            # Update the best solution
            idx_child = np.argmin(fitness + fitness_child)
            if fitness_child[idx_child] < self.f_best:
                self.f_best = fitness_child[idx_child]
                self.x_best = child

            # Store the history
            self.f_history.append(self.f_best)
            self.x_history.append(self.x_best)

# Example usage:
def func(x):
    return np.sum(x**2)

cdea = CDEA(budget=100, dim=5)
cdea(0)(func)