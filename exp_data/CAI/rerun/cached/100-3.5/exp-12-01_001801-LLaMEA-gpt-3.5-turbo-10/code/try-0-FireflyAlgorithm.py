import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim, alpha=0.2, beta0=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

    def __call__(self, func):
        def attractiveness(r):
            return self.beta0 * np.exp(-self.gamma * r**2)

        def distance(x, y):
            return np.linalg.norm(x - y)

        def levy_flight(dim):
            sigma = (np.gamma(1 + self.alpha) * np.sin(np.pi * self.alpha / 2) / np.gamma((1 + self.alpha) / 2) / 2**((self.alpha - 1) / 2))**(1 / self.alpha)
            u = np.random.normal(0, sigma, dim)
            v = np.random.normal(0, 1, dim)
            step = u / np.abs(v)**(1 / self.alpha)
            return step

        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)

        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if fitness[i] < fitness[j]:
                        r = distance(population[i], population[j])
                        beta = attractiveness(r)
                        population[i] += beta * levy_flight(self.dim) + np.random.normal(0, 1, self.dim)
                        population[i] = np.clip(population[i], -5.0, 5.0)
                        fitness[i] = func(population[i])
                        if fitness[i] < fitness[best_idx]:
                            best_idx = i

        return population[best_idx]