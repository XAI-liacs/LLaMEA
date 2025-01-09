import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim, alpha=0.5, beta0=1.0, gamma=0.1):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

    def attractiveness(self, light_intensity, distance):
        return self.beta0 * np.exp(-self.gamma * distance**2) * light_intensity

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))
        light_intensity = np.array([func(x) for x in population])
        best_solution = population[np.argmin(light_intensity)]
        
        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if light_intensity[j] < light_intensity[i]:
                        distance = np.linalg.norm(population[i] - population[j])
                        population[i] += self.alpha * (population[j] - population[i]) + self.attractiveness(light_intensity[j], distance) * np.random.uniform(-1, 1, size=self.dim)
                        light_intensity[i] = func(population[i])
                        
            best_solution = population[np.argmin(light_intensity)]
        
        return best_solution