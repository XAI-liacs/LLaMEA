import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta0=1.0, gamma=0.01):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, r):
        return self.beta0 * np.exp(-self.gamma * r**2)

    def move_fireflies(self, fireflies, func):
        for i in range(len(fireflies)):
            for j in range(len(fireflies)):
                if func(fireflies[j]) < func(fireflies[i]):
                    r = np.linalg.norm(fireflies[j] - fireflies[i])
                    beta = self.attractiveness(r)
                    fireflies[i] = fireflies[i] + beta * (fireflies[j] - fireflies[i]) + self.alpha * np.random.uniform(-1, 1, self.dim)
        return fireflies

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for _ in range(self.budget):
            fireflies = self.move_fireflies(fireflies, func)
        
        idx = np.argmin([func(f) for f in fireflies])
        self.f_opt = func(fireflies[idx])
        self.x_opt = fireflies[idx]
        
        return self.f_opt, self.x_opt