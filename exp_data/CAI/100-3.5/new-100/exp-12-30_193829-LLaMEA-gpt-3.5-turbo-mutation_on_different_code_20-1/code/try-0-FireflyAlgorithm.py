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

    def attractiveness(self, x, y):
        return np.exp(-self.gamma * np.linalg.norm(x - y)**2)

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        
        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(population[j]) < func(population[i]):
                        beta = self.beta0 * np.exp(-self.alpha * np.linalg.norm(population[i] - population[j])**2)
                        population[i] += beta * (population[j] - population[i]) + np.random.uniform(-1, 1, self.dim)
            
            for i in range(self.budget):
                f = func(population[i])
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = population[i]
        
        return self.f_opt, self.x_opt