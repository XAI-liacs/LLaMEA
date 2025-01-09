import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim, alpha=0.2, beta=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __call__(self, func):
        n_fireflies = self.budget
        firefly_positions = np.random.uniform(-5.0, 5.0, (n_fireflies, self.dim))
        intensities = np.array([func(x) for x in firefly_positions])
        
        for _ in range(self.budget):
            for i in range(n_fireflies):
                for j in range(n_fireflies):
                    if intensities[j] < intensities[i]:
                        r = np.linalg.norm(firefly_positions[i] - firefly_positions[j])
                        beta_mult = self.beta * np.exp(-self.gamma * r**2)
                        firefly_positions[i] = firefly_positions[i] + self.alpha * (firefly_positions[j] - firefly_positions[i]) + beta_mult * np.random.uniform(-1, 1, self.dim)
                        firefly_positions[i] = np.clip(firefly_positions[i], -5.0, 5.0)
                        intensities[i] = func(firefly_positions[i])
        
        best_pos = firefly_positions[np.argmin(intensities)]
        return best_pos