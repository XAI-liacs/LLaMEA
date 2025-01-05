import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta0=1.0, gamma=0.1):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, x_i, x_j):
        return self.beta0 * np.exp(-self.gamma * np.linalg.norm(x_i - x_j))

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))

        for i in range(self.budget):
            for j in range(self.budget):
                if func(population[j]) < func(population[i]):
                    attractiveness_ij = self.attractiveness(population[i], population[j])
                    population[i] += self.alpha * (population[j] - population[i]) * attractiveness_ij

            f = func(population[i])
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = population[i]
        
        return self.f_opt, self.x_opt