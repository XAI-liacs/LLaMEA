import numpy as np
from scipy.spatial.distance import cdist

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, r):
        return np.exp(-self.beta * r**2)

    def move_fireflies(self, fireflies, func):
        for i in range(len(fireflies)):
            for j in range(len(fireflies)):
                if func(fireflies[j]) < func(fireflies[i]):
                    r = np.linalg.norm(fireflies[j] - fireflies[i])
                    beta_ij = self.attractiveness(r)
                    fireflies[i] += self.alpha * (fireflies[j] - fireflies[i]) * beta_ij

        return fireflies

    def __call__(self, func):
        fireflies = np.random.uniform(low=-5.0, high=5.0, size=(self.budget, self.dim))

        for _ in range(self.budget):
            fireflies = self.move_fireflies(fireflies, func)

        f_values = np.array([func(f) for f in fireflies])
        min_idx = np.argmin(f_values)
        
        self.f_opt = f_values[min_idx]
        self.x_opt = fireflies[min_idx]

        return self.f_opt, self.x_opt