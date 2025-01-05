import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta_min=0.2, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta_min = beta_min
        self.gamma = gamma
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, xi, xj):
        return np.exp(-self.gamma * np.linalg.norm(xi - xj))

    def move_firefly(self, x, best_x, i):
        beta = self.beta_min * np.exp(-self.alpha * i)
        x += beta * (best_x - x) + 0.01 * np.random.normal(0, 1, self.dim)
        return np.clip(x, -5.0, 5.0)

    def update_alpha(self, t):
        return self.alpha * 0.95 ** t

    def __call__(self, func):
        fireflies = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        intensities = np.zeros(self.budget)
        
        for t in range(self.budget):
            self.alpha = self.update_alpha(t)
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(fireflies[j]) < func(fireflies[i]):
                        intensities[i] += self.attractiveness(fireflies[i], fireflies[j])

                if intensities[i] > self.f_opt:
                    self.f_opt = intensities[i]
                    self.x_opt = fireflies[i]

                fireflies[i] = self.move_firefly(fireflies[i], self.x_opt, t)
        
        return func(self.x_opt), self.x_opt