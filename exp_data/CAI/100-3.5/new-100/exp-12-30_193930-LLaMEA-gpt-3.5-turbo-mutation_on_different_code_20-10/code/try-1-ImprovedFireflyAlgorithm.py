class ImprovedFireflyAlgorithm(FireflyAlgorithm):
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta0=1.0, gamma=0.01):
        super().__init__(budget, dim, alpha, beta0, gamma)
        self.alpha_min = 0.2

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))

        for i in range(self.budget):
            for j in range(self.budget):
                if func(fireflies[j]) < func(fireflies[i]):
                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    beta = self.beta0 * np.exp(-self.alpha * r**2)
                    fireflies[i] = fireflies[i] + beta * (fireflies[j] - fireflies[i]) + 0.01 * np.random.normal(size=self.dim)
                    self.alpha = max(self.alpha_min, self.alpha * 0.99)  # Dynamic adaptation

            f = func(fireflies[i])
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = fireflies[i]

        return self.f_opt, self.x_opt