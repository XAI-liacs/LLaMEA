import numpy as np

class QuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def quantum_mutation(self, x, sigma):
        return x + np.random.normal(0, sigma, size=self.dim)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        sigma = 0.5
        for _ in range(self.budget):
            offspring = np.array([self.quantum_mutation(individual, sigma) for individual in population])
            population = np.where([func(offspring[i]) < func(population[i]) for i in range(self.budget)], offspring, population)
            sigma *= 0.95  # Decrease mutation step size
        return population[np.argmin([func(individual) for individual in population])]