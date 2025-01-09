import numpy as np

class FireflyOptimization:
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta_min=0.2, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta_min = beta_min
        self.gamma = gamma
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, x, y):
        return np.exp(-self.gamma * np.linalg.norm(x - y)**2)

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        
        for i in range(self.budget):
            for j in range(self.budget):
                if func(population[j]) < func(population[i]):
                    beta = self.beta_min + (1 - self.beta_min) * np.exp(-self.alpha * np.linalg.norm(population[j] - population[i])**2)
                    population[i] += beta * (population[j] - population[i]) + np.random.uniform(-1, 1, self.dim)
                    
            if func(population[i]) < self.f_opt:
                self.f_opt = func(population[i])
                self.x_opt = population[i]
                
        return self.f_opt, self.x_opt