import numpy as np

class SpiralDynamicsOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        best_solution = None
        best_fitness = np.inf

        for i in range(self.budget):
            for j in range(self.dim):
                population[i][j] += np.sin(i) * np.cos(j)  # Update based on Spiral Dynamics

            fitness = func(population[i])
            if fitness < best_fitness:
                best_solution = population[i]
                best_fitness = fitness

        return best_solution