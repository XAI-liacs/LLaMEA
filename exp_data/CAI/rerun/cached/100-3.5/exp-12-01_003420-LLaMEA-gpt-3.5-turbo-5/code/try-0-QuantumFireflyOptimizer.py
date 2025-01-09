import numpy as np

class QuantumFireflyOptimizer:
    def __init__(self, budget, dim, num_fireflies=20, alpha=0.2, beta0=1.0, gamma=0.1):
        self.budget = budget
        self.dim = dim
        self.num_fireflies = num_fireflies
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

    def __call__(self, func):
        def attractiveness(beta, r):
            return beta * np.exp(-self.gamma * r**2)

        def levy_flight():
            sigma = (np.gamma(1 + self.alpha) * np.sin(np.pi * self.alpha / 2) / np.pi) ** (1 / self.alpha)
            u = np.random.normal(0, sigma, self.dim)
            v = np.random.normal(0, 1, self.dim)
            step = u / np.abs(v) ** (1 / self.alpha)
            return step

        def initialize_fireflies():
            return np.random.uniform(-5.0, 5.0, size=(self.num_fireflies, self.dim))

        fireflies = initialize_fireflies()
        best_firefly = fireflies[np.argmin([func(x) for x in fireflies])]
        for _ in range(self.budget):
            for i in range(self.num_fireflies):
                for j in range(self.num_fireflies):
                    if func(fireflies[i]) < func(fireflies[j]):
                        r = np.linalg.norm(fireflies[i] - fireflies[j])
                        beta = self.beta0 * np.exp(-self.gamma * r**2)
                        step = levy_flight()
                        fireflies[i] += attractiveness(beta, r) * (fireflies[j] - fireflies[i]) + step
                fireflies[i] = np.clip(fireflies[i], -5.0, 5.0)
                if func(fireflies[i]) < func(best_firefly):
                    best_firefly = fireflies[i]
        return best_firefly