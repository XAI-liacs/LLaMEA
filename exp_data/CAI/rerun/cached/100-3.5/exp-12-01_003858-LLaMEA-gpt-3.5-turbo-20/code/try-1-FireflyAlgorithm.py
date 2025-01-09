import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim, alpha=0.2, beta0=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

    def attractiveness(self, i, j):
        return self.beta0 * np.exp(-self.gamma * np.linalg.norm(i - j))

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if fitness[j] < fitness[i]:
                        attractiveness_ij = self.attractiveness(population[i], population[j])
                        population[i] += self.alpha * (population[j] - population[i]) * attractiveness_ij
                        fitness[i] = func(population[i])
        
        best_idx = np.argmin(fitness)
        return population[best_idx]