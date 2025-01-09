class ImprovedFireflyAlgorithm(FireflyAlgorithm):
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta=1.0, gamma=1.0, alpha_decay=0.95, beta_growth=1.05):
        super().__init__(budget, dim, alpha, beta, gamma)
        self.alpha_decay = alpha_decay
        self.beta_growth = beta_growth

    def move_fireflies(self, fireflies, func):
        for i in range(len(fireflies)):
            for j in range(len(fireflies)):
                if func(fireflies[i]) < func(fireflies[j]):
                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    attr = self.attractiveness(r)
                    self.alpha *= self.alpha_decay
                    self.beta *= self.beta_growth
                    fireflies[i] += attr * (fireflies[j] - fireflies[i]) + self.alpha * (np.random.rand(self.dim) - 0.5)

        return fireflies