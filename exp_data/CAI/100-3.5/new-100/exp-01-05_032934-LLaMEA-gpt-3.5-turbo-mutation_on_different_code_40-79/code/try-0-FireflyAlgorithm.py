import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta_min=0.2, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta_min = beta_min
        self.gamma = gamma
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, r):
        return 1.0 / (1.0 + self.alpha * r**2)

    def move_fireflies(self, fireflies, func):
        for i in range(len(fireflies)):
            for j in range(len(fireflies)):
                if func(fireflies[j]) < func(fireflies[i]):
                    beta = self.beta_min + (1.0 - self.beta_min) * np.exp(-self.gamma * np.linalg.norm(fireflies[j] - fireflies[i])**2)
                    fireflies[i] += beta * (fireflies[j] - fireflies[i]) + 0.01 * np.random.normal(0, 1, self.dim)

        return fireflies

    def __call__(self, func):
        fireflies = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        
        for i in range(self.budget):
            fireflies = self.move_fireflies(fireflies, func)
            
            f_values = np.array([func(f) for f in fireflies])
            idx = np.argmin(f_values)
            if f_values[idx] < self.f_opt:
                self.f_opt = f_values[idx]
                self.x_opt = fireflies[idx]
            
        return self.f_opt, self.x_opt