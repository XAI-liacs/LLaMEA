import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta0=1.0, gamma=0.1):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        def attractiveness(beta, r):
            return beta * np.exp(-self.gamma * r**2)

        def update_firefly(x_i, x_j, beta):
            r = np.linalg.norm(x_i - x_j)
            return x_i + attractiveness(beta, r) * (x_j - x_i) + self.alpha * np.random.normal(size=self.dim)

        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            for j in range(self.budget):
                if func(population[j]) < func(population[i]):
                    population[i] = update_firefly(population[i], population[j], self.beta0)

            f = func(population[i])
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = population[i]

        return self.f_opt, self.x_opt