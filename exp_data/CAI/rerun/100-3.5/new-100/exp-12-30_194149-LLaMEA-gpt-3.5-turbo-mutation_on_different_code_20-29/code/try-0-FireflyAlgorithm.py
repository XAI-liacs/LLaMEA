import numpy as np
from scipy.spatial.distance import cdist

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, r):
        return self.beta * np.exp(-self.gamma * r**2)

    def move_fireflies(self, fireflies, func):
        for i in range(len(fireflies)):
            for j in range(len(fireflies)):
                if func(fireflies[i]) < func(fireflies[j]):
                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    attr = self.attractiveness(r)
                    fireflies[i] += attr * (fireflies[j] - fireflies[i]) + self.alpha * (np.random.rand(self.dim) - 0.5)

        return fireflies

    def __call__(self, func):
        fireflies = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))

        for _ in range(self.budget):
            fireflies = self.move_fireflies(fireflies, func)

        f_values = np.array([func(f) for f in fireflies])
        min_idx = np.argmin(f_values)
        self.f_opt = f_values[min_idx]
        self.x_opt = fireflies[min_idx]

        return self.f_opt, self.x_opt