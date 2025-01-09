import numpy as np

class QIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            fitness = [func(ind) for ind in population]
            best_index = np.argmin(fitness)
            quantum_mutation = np.random.uniform(-1.0, 1.0, self.dim) * (population[best_index] - population)
            population += quantum_mutation
        return population[best_index]