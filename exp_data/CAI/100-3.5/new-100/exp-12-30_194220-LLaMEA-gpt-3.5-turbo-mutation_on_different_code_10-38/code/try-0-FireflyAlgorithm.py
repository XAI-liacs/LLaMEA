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

    def attractiveness(self, r):
        return self.beta0 * np.exp(-self.gamma * r**2)

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for i in range(self.budget):
            for j in range(self.budget):
                if fitness[j] < fitness[i]:
                    r = np.linalg.norm(population[i] - population[j])
                    beta = self.attractiveness(r)
                    population[i] += self.alpha * (population[j] - population[i]) * beta
                    fitness[i] = func(population[i])
                    
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]
        
        return self.f_opt, self.x_opt