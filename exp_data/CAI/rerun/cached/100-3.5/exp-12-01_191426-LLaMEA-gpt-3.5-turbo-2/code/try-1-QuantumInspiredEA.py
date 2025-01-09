import numpy as np

class QuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def quantum_mutation(self, x, sigma):
        return x + np.random.normal(0, sigma, size=self.dim)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        sigma_base = 0.5
        for _ in range(self.budget):
            offspring = np.array([self.quantum_mutation(individual, sigma_base) for individual in population])
            improved_offspring = np.where([func(offspring[i]) < func(population[i]) for i in range(self.budget)], offspring, population)
            fitness_improvement = np.mean([func(offspring[i]) - func(population[i]) for i in range(self.budget)])
            sigma = sigma_base * (1 + fitness_improvement)  # Dynamic mutation step size adjustment
            population = improved_offspring
            sigma_base = max(0.1, min(0.5, sigma))  # Bound sigma between 0.1 and 0.5
        return population[np.argmin([func(individual) for individual in population])]