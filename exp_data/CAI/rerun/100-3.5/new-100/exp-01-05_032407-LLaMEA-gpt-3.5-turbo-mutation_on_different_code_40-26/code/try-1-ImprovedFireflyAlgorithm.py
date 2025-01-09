class ImprovedFireflyAlgorithm(FireflyAlgorithm):
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta0=1.0, gamma=1.0, delta=0.1):
        super().__init__(budget, dim, alpha, beta0, gamma)
        self.delta = delta

    def light_intensity(self, f):
        return 1 / (1 + f)

    def move_firefly(self, x_i, x_j, f_i, f_j):
        beta = self.attractiveness(x_i, x_j) * self.light_intensity(f_i) * self.light_intensity(f_j)
        x_new = x_i + beta * (x_j - x_i) + self.alpha * (np.random.rand(self.dim) - 0.5) * self.delta
        return x_new