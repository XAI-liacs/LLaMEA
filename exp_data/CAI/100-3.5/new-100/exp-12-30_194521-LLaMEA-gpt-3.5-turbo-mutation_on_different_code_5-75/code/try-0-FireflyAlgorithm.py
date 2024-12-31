import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta0=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, x, y):
        return self.beta0 * np.exp(-self.gamma * np.linalg.norm(x - y))

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(population[j]) < func(population[i]):
                        population[i] += self.alpha * (population[j] - population[i]) + self.attractiveness(population[i], population[j]) * np.random.uniform(-1, 1, size=self.dim)
            
        best_idx = np.argmin([func(individual) for individual in population])
        self.f_opt = func(population[best_idx])
        self.x_opt = population[best_idx]
        
        return self.f_opt, self.x_opt