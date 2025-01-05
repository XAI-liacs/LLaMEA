import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta0=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.fitness = np.array([np.Inf] * budget)

    def attractiveness(self, r):
        return self.beta0 * np.exp(-self.alpha * r**2)

    def move_firefly(self, source, target):
        r = np.linalg.norm(source - target)
        beta = self.attractiveness(r)
        return target + beta * (source - target) + 0.01 * np.random.randn(self.dim)

    def __call__(self, func):
        for i in range(self.budget):
            for j in range(self.budget):
                if func(self.population[i]) < func(self.population[j]):
                    self.population[i] = self.move_firefly(self.population[i], self.population[j])

        idx = np.argmin([func(x) for x in self.population])
        return func(self.population[idx]), self.population[idx]