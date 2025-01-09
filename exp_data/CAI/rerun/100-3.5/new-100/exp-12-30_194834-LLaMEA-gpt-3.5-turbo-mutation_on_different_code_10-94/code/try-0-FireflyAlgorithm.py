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

    def attractiveness(self, r):
        return self.beta0 * np.exp(-self.gamma * r**2)

    def __call__(self, func):
        fireflies = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))
        for i in range(self.budget):
            for j in range(self.budget):
                if func(fireflies[j]) < func(fireflies[i]):
                    r = np.linalg.norm(fireflies[j] - fireflies[i])
                    beta = self.attractiveness(r)
                    fireflies[i] += self.alpha * (fireflies[j] - fireflies[i]) * beta

        best_idx = np.argmin([func(firefly) for firefly in fireflies])
        self.f_opt = func(fireflies[best_idx])
        self.x_opt = fireflies[best_idx]

        return self.f_opt, self.x_opt