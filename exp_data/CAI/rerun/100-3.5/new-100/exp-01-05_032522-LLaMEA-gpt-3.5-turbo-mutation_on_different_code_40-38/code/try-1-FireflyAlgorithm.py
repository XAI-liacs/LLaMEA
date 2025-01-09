class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta0=1.0, gamma=0.2):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.fitness = np.array([np.Inf] * budget)
    
    def attractiveness(self, r):
        return self.beta0 * np.exp(-self.alpha * r**2)
    
    def move_fireflies(self, func):
        for i in range(self.budget):
            for j in range(self.budget):
                if self.fitness[i] > self.fitness[j]:
                    r = np.linalg.norm(self.population[i] - self.population[j])
                    beta = self.attractiveness(r)
                    step_size = self.gamma * np.linalg.norm(self.population[i] - self.population[j])
                    self.population[i] += beta * (self.population[j] - self.population[i]) + step_size * np.random.uniform(-1, 1, size=self.dim)
                    self.population[i] = np.clip(self.population[i], -5.0, 5.0)
                    self.fitness[i] = func(self.population[i])
    
    def __call__(self, func):
        for i in range(self.budget):
            self.fitness[i] = func(self.population[i])
        
        for _ in range(self.budget):
            self.move_fireflies(func)
        
        best_idx = np.argmin(self.fitness)
        return self.fitness[best_idx], self.population[best_idx]