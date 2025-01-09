import numpy as np

class FireflySwarmOptimization:
    def __init__(self, budget, dim, alpha=0.2, beta0=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

    def attractiveness(self, brightness_i, brightness_j, distance):
        return self.beta0 * np.exp(-self.gamma * distance**2) * brightness_i

    def __call__(self, func):
        def initialize_fireflies():
            return np.random.uniform(-5.0, 5.0, (self.budget, self.dim))

        def optimize():
            fireflies = initialize_fireflies()
            best_position = fireflies[np.argmin([func(firefly) for firefly in fireflies])]
            for _ in range(self.budget):
                for i in range(self.budget):
                    for j in range(self.budget):
                        if func(fireflies[j]) < func(fireflies[i]):
                            distance = np.linalg.norm(fireflies[i] - fireflies[j])
                            attractiveness_value = self.attractiveness(func(fireflies[i]), func(fireflies[j]), distance)
                            fireflies[i] += self.alpha * (fireflies[j] - fireflies[i]) * attractiveness_value
                best_position = fireflies[np.argmin([func(firefly) for firefly in fireflies])]
            return best_position

        return optimize()