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

    def move_firefly(self, x_i, x_j, attractiveness):
        r = np.linalg.norm(x_i - x_j)
        beta = self.attractiveness(r)
        x_i = x_i + beta * (x_j - x_i) + self.alpha * (np.random.rand(self.dim) - 0.5)
        return x_i

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))

        for i in range(self.budget):
            for j in range(i+1, self.budget):
                if func(fireflies[i]) < func(fireflies[j]):
                    fireflies[i] = self.move_firefly(fireflies[i], fireflies[j], self.attractiveness(np.linalg.norm(fireflies[i] - fireflies[j])))
                else:
                    fireflies[j] = self.move_firefly(fireflies[j], fireflies[i], self.attractiveness(np.linalg.norm(fireflies[i] - fireflies[j])))

        self.x_opt = fireflies[np.argmin([func(x) for x in fireflies])]
        self.f_opt = func(self.x_opt)
        
        return self.f_opt, self.x_opt