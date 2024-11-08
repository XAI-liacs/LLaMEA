import numpy as np

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            fitness_vals = [func(ind) for ind in population]
            elites = population[np.argsort(fitness_vals)[:2]]
            offspring = np.mean(elites, axis=0) + np.random.normal(0, 1, self.dim)
            replace_idx = np.argmax(fitness_vals)
            population[replace_idx] = offspring
        return population[np.argmin(fitness_vals)]