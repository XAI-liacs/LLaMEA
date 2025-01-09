import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.alpha = 0.2
        self.beta = 1.0

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget):
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if fitness[j] < fitness[i]:
                        attractiveness = np.exp(-self.beta * np.linalg.norm(population[i] - population[j]))
                        population[i] += self.alpha * (population[j] - population[i]) * attractiveness
                        population[i] = np.clip(population[i], -5.0, 5.0)
                        fitness[i] = func(population[i])
        
        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        return best_solution