import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim, alpha=0.2, beta0=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))

        def attractiveness(distance, beta):
            return beta * np.exp(-self.gamma * distance**2)

        population = initialize_population()
        fitness = np.array([func(individual) for individual in population])
        best_idx = np.argmin(fitness)

        for _ in range(self.budget):
            beta = self.beta0 * np.exp(-self.gamma * _ / self.budget)
            
            for i in range(self.budget):
                for j in range(self.budget):
                    if fitness[i] > fitness[j]:
                        distance = np.linalg.norm(population[i] - population[j])
                        epsilon = self.alpha * np.random.uniform(-1, 1, size=self.dim)
                        population[i] += attractiveness(distance, beta) * (population[j] - population[i]) + epsilon
                        population[i] = np.clip(population[i], -5.0, 5.0)
                        fitness[i] = func(population[i])

            new_best_idx = np.argmin(fitness)
            if fitness[new_best_idx] < fitness[best_idx]:
                best_idx = new_best_idx

        return population[best_idx]