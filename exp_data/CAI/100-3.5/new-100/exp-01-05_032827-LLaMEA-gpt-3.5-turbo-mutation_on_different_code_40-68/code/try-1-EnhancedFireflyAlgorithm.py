class EnhancedFireflyAlgorithm(FireflyAlgorithm):
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta_min=0.2, gamma=1.0, gamma_min=0.1, gamma_max=2.0):
        super().__init__(budget, dim, alpha, beta_min, gamma)
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def attractiveness(self, r):
        return np.exp(-self.gamma * r**2)

    def move_firefly(self, x, x_best):
        r = np.linalg.norm(x - x_best)
        self.gamma = max(self.gamma_min, min(self.gamma_max, self.gamma * 0.9))
        beta = self.beta_min * np.exp(-self.alpha * r**2)
        x = x + beta * (x_best - x) + 0.01 * np.random.uniform(-1, 1, size=self.dim)
        return np.clip(x, -5.0, 5.0)