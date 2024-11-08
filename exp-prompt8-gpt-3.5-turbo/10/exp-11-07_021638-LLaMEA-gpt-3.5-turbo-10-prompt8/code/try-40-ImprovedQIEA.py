import numpy as np

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            fitness_values = [func(ind) for ind in population]
            elites = population[np.argsort(fitness_values)[:2]]
            offspring = elites.mean(axis=0) + np.random.normal(0, 1, self.dim)
            worst_idx = np.argmax(fitness_values)
            population[worst_idx] = offspring
        return population[np.argmin([func(ind) for ind in population])]