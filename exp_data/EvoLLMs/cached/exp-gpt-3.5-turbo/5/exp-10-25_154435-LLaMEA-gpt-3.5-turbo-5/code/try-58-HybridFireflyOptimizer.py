import numpy as np

class HybridFireflyOptimizer:
    def __init__(self, budget, dim, population_size=50, alpha=0.2, beta0=1.0, gamma=0.01):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        best_solution = population[np.argmin([func(individual) for individual in population])

        for _ in range(self.budget):
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if func(population[j]) < func(population[i]):
                        distance = np.linalg.norm(population[i] - population[j])
                        attractiveness = self.beta0 * np.exp(-self.gamma * distance ** 2)
                        step = self.alpha * (np.random.rand(self.dim) - 0.5)
                        population[i] += attractiveness * (population[j] - population[i]) + step

            best_solution = population[np.argmin([func(individual) for individual in population])]

        return best_solution