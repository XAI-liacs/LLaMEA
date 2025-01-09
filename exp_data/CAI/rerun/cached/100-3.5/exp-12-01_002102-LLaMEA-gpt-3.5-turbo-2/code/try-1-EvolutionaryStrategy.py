import numpy as np

class EvolutionaryStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def __call__(self, func):
        for _ in range(self.budget):
            offspring = self.population + np.random.normal(0, 1, (self.budget, self.dim))
            scores = [func(individual) for individual in offspring]
            self.population = offspring[np.argsort(scores)]

        return self.population[0]