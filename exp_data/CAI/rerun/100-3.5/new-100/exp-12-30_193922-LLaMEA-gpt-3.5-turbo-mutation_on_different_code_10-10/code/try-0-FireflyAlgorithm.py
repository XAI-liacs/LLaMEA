import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.light_intensity = np.zeros(budget)
    
    def attractiveness(self, r):
        return self.alpha * np.exp(-self.gamma * r**2)
    
    def move_firefly(self, current, target):
        r = np.linalg.norm(current - target)
        new_position = current + self.attractiveness(r) * (target - current) + self.beta * np.random.uniform(-1, 1, self.dim)
        return np.clip(new_position, -5.0, 5.0)
    
    def __call__(self, func):
        for i in range(self.budget):
            for j in range(self.budget):
                if func(self.population[j]) < func(self.population[i]):
                    self.population[i] = self.move_firefly(self.population[i], self.population[j])
        
        best_index = np.argmin([func(x) for x in self.population])
        self.f_opt = func(self.population[best_index])
        self.x_opt = self.population[best_index]
        
        return self.f_opt, self.x_opt