import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta0=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, xi, xj):
        return np.exp(-self.gamma * np.linalg.norm(xi - xj) ** 2)

    def move_firefly(self, x, best_x, beta):
        epsilon = np.random.uniform(-1, 1, size=self.dim)
        return x + beta * (best_x - x) + self.alpha * epsilon

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        
        for i in range(self.budget):
            for j in range(self.budget):
                if func(fireflies[j]) < func(fireflies[i]):
                    attractiveness_ij = self.attractiveness(fireflies[i], fireflies[j])
                    beta = self.beta0 * np.exp(-self.alpha * i / self.budget)
                    fireflies[i] = self.move_firefly(fireflies[i], fireflies[j], beta) + attractiveness_ij

        self.x_opt = fireflies[np.argmin([func(x) for x in fireflies])]
        self.f_opt = func(self.x_opt)
        
        return self.f_opt, self.x_opt