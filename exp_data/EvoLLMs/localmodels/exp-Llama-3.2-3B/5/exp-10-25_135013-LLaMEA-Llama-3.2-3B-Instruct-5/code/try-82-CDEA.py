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
        self.mutation_prob = 0.05
        self.crossover_prob = 0.05

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

        # Perform selection
        selected_population = population[np.argsort(fitness)]

        # Perform crossover
        new_population = []
        for i in range(self.budget):
            if random.random() < self.crossover_prob:
                parent1 = selected_population[i]
                parent2 = selected_population[(i+1) % self.budget]
                child = (parent1 + parent2) / 2
                new_population.append(child)
            else:
                new_population.append(selected_population[i])

        # Perform mutation
        for i in range(self.budget):
            if random.random() < self.mutation_prob:
                child = new_population[i]
                if random.random() < 0.5:
                    child += np.random.uniform(-1.0, 1.0, size=self.dim)
                new_population[i] = child

        # Evaluate the new population
        fitness_new = [func(x) for x in new_population]

        # Update the best solution
        idx_new = np.argmin(fitness_new)
        if fitness_new[idx_new] < self.f_best:
            self.f_best = fitness_new[idx_new]
            self.x_best = new_population[idx_new]

        # Store the history
        self.f_history.append(self.f_best)
        self.x_history.append(self.x_best)

# Example usage:
def func(x):
    return np.sum(x**2)

cdea = CDEA(budget=100, dim=5)
cdea(0)(func)