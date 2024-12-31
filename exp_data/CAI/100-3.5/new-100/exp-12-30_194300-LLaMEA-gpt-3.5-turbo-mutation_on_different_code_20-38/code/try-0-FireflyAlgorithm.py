import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, r):
        return self.alpha * np.exp(-self.beta * r**2)

    def move_firefly(self, x_i, x_j, attractiveness):
        r = np.linalg.norm(x_i - x_j)
        e = (x_j - x_i) * attractiveness + self.gamma * (np.random.rand(self.dim) - 0.5)
        return x_i + e

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        intensities = np.array([func(x) for x in fireflies])

        for i in range(self.budget):
            for j in range(self.budget):
                if intensities[i] > intensities[j]:
                    attractiveness_ij = self.attractiveness(np.linalg.norm(fireflies[i] - fireflies[j]))
                    fireflies[i] = self.move_firefly(fireflies[i], fireflies[j], attractiveness_ij)
                    intensities[i] = func(fireflies[i])

        idx_opt = np.argmin(intensities)
        self.f_opt = intensities[idx_opt]
        self.x_opt = fireflies[idx_opt]

        return self.f_opt, self.x_opt