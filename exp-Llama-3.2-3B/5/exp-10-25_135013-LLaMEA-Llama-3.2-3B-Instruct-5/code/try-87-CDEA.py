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
        self.mutation_prob = 0.2

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
            if random.random() < self.mutation_prob:
                child += np.random.uniform(-1.0, 1.0, size=self.dim)

            # Adjust mutation probability based on fitness
            if fitness[idx_child] < self.f_best:
                self.mutation_prob *= 0.99
            elif fitness[idx_child] > self.f_best:
                self.mutation_prob *= 1.01

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

            # Adjust mutation probability
            self.mutation_prob = np.clip(self.mutation_prob, 0.01, 0.3)

# Example usage:
def func(x):
    return np.sum(x**2)

cdea = CDEA(budget=100, dim=5)
cdea(0)(func)