import numpy as np
import random

class DEASPS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.adaptive_step_size = 0.5
        self.adaptive_population_size = 0.5
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.line_search_probability = 0.1

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        for i in range(self.budget):
            fitness = np.array([func(x) for x in population])
            best_individual = population[np.argmin(fitness)]
            new_population = population + self.adaptive_step_size * np.random.uniform(-1, 1, (self.population_size, self.dim))
            if np.random.rand() < self.adaptive_population_size:
                new_population = new_population[:int(self.population_size * self.adaptive_population_size)]
            if np.random.rand() < self.line_search_probability:
                new_population = self.line_search(new_population, func, fitness, best_individual)
            population[np.argsort(fitness)] = np.concatenate((population[np.argsort(fitness)], new_population[np.argsort(fitness)]))
        return population[np.argmin(fitness)]

    def line_search(self, new_population, func, fitness, best_individual):
        new_fitness = np.array([func(x) for x in new_population])
        best_index = np.argmin(new_fitness)
        best_new_individual = new_population[best_index]
        step_size = 0.01
        while True:
            new_population[best_index] += step_size * (best_individual - best_new_individual)
            new_fitness[best_index] = func(new_population[best_index])
            if new_fitness[best_index] < fitness[best_index]:
                return new_population
            step_size *= 0.9

# Example usage:
def func(x):
    return np.sum(x**2)

de = DEASPS(budget=100, dim=10)
best_x = de(func)
print("Best x:", best_x)
print("Best f(x):", func(best_x))