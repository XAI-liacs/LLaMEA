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

    def attractiveness(self, xi, xj):
        return self.beta0 * np.exp(-self.gamma * np.linalg.norm(xi - xj))

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))

        for i in range(self.budget):
            for j in range(self.budget):
                if func(fireflies[j]) < func(fireflies[i]):
                    attractiveness_ij = self.attractiveness(fireflies[i], fireflies[j])
                    fireflies[i] += self.alpha * attractiveness_ij * (fireflies[j] - fireflies[i])

        best_idx = np.argmin([func(f) for f in fireflies])
        self.f_opt = func(fireflies[best_idx])
        self.x_opt = fireflies[best_idx]

        return self.f_opt, self.x_opt