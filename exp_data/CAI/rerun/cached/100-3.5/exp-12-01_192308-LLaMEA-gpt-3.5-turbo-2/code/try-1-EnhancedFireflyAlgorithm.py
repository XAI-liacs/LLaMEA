import numpy as np

class EnhancedFireflyAlgorithm(FireflyAlgorithm):
    def __init__(self, budget, dim, alpha=0.2, beta=1.0, gamma=1.0, levy_scale=0.1):
        super().__init__(budget, dim, alpha, beta, gamma)
        self.levy_scale = levy_scale

    def levy_flight(self, dim):
        sigma = (np.math.gamma(1 + self.levy_scale) * np.sin(np.pi * self.levy_scale / 2) / np.math.gamma((1 + self.levy_scale) / 2) * self.levy_scale ** ((1 + self.levy_scale) / 2)) ** (1 / self.levy_scale)
        u = np.random.normal(0, sigma, dim)
        v = np.random.normal(0, 1, dim)
        step = u / abs(v) ** (1 / self.levy_scale)
        return step

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
                        levy_step = self.levy_scale * self.levy_flight(self.dim)
                        firefly_positions[i] = firefly_positions[i] + self.alpha * (firefly_positions[j] - firefly_positions[i]) + beta_mult * np.random.uniform(-1, 1, self.dim) + levy_step
                        firefly_positions[i] = np.clip(firefly_positions[i], -5.0, 5.0)
                        intensities[i] = func(firefly_positions[i])
        
        best_pos = firefly_positions[np.argmin(intensities)]
        return best_pos