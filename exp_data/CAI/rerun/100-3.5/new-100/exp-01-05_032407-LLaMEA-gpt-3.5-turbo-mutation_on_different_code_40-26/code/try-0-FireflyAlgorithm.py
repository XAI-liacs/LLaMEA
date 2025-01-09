import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta0=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, x_i, x_j):
        return self.beta0 * np.exp(-self.gamma * np.linalg.norm(x_i - x_j))

    def move_firefly(self, x_i, x_j):
        beta = self.attractiveness(x_i, x_j)
        x_new = x_i + beta * (x_j - x_i) + self.alpha * (np.random.rand(self.dim) - 0.5)
        return x_new

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))

        for i in range(self.budget):
            for j in range(self.budget):
                if func(population[j]) < func(population[i]):
                    population[i] = self.move_firefly(population[i], population[j])

        best_idx = np.argmin([func(x) for x in population])
        self.f_opt = func(population[best_idx])
        self.x_opt = population[best_idx]

        return self.f_opt, self.x_opt