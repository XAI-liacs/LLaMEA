import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.alpha = 0.2
        self.beta0 = 1.0
        self.gamma = 0.2
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, x, y):
        return self.beta0 * np.exp(-self.gamma * np.linalg.norm(x - y))

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.population_size, self.dim))
        
        for _ in range(self.budget):
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if func(population[j]) < func(population[i]):
                        attractiveness_ij = self.attractiveness(population[i], population[j])
                        population[i] += attractiveness_ij * (population[j] - population[i]) + self.alpha * np.random.normal(size=self.dim)
            
            f_values = [func(x) for x in population]
            min_idx = np.argmin(f_values)
            if f_values[min_idx] < self.f_opt:
                self.f_opt = f_values[min_idx]
                self.x_opt = population[min_idx].copy()
        
        return self.f_opt, self.x_opt