import numpy as np

class ImprovedFireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta0=1.0, gamma=1.0, delta=0.1):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.delta = delta
        self.f_opt = np.Inf
        self.x_opt = None

    def levy_flight(self):
        sigma = (np.math.gamma(1 + self.beta0) * np.math.sin(np.pi * self.beta0 / 2) / (np.math.gamma((1 + self.beta0) / 2) * self.beta0 * 2 ** ((self.beta0 - 1) / 2))) ** (1 / self.beta0)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v) ** (1 / self.beta0)
        return step

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        intensities = np.array([func(x) for x in population])
        
        for _ in range(self.budget):
            for i in range(self.budget):
                attractiveness_i = intensities[i]
                for j in range(self.budget):
                    attractiveness_j = intensities[j]
                    if attractiveness_i > attractiveness_j:
                        r = np.linalg.norm(population[i] - population[j])
                        beta = self.beta0 * np.exp(-self.gamma * r**2)
                        step = self.levy_flight()
                        population[i] += self.alpha * np.exp(-beta) * (population[j] - population[i]) + step
                        population[i] = np.clip(population[i], func.bounds.lb, func.bounds.ub)
                        intensities[i] = func(population[i])
                        if intensities[i] < self.f_opt:
                            self.f_opt = intensities[i]
                            self.x_opt = population[i]
                
                population[i] += self.delta * np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)  # Introducing diversity
        
        return self.f_opt, self.x_opt