import numpy as np

class QIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.max_gen = budget // self.population_size

    def __call__(self, func):
        def quantum_mutation(x, sigma):
            return x + sigma * np.random.normal(0, 1, size=len(x))

        population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        sigma = 0.2

        for _ in range(self.max_gen):
            offspring = np.array([quantum_mutation(individual, sigma) for individual in population])
            fitness_values = np.array([func(individual) for individual in offspring])
            sorted_indices = np.argsort(fitness_values)
            population = offspring[sorted_indices[:self.population_size]]
            sigma *= 0.9

        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution