class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.alpha = 0.5
        self.beta0 = 1.0
        self.gamma = 0.01

    def attractiveness(self, x_i, x_j):
        return self.beta0 * np.exp(-self.gamma * np.linalg.norm(x_i - x_j) ** 2)

    def move_firefly(self, x_i, x_j, alpha):
        r = np.random.uniform(-1, 1, size=self.dim)
        x_new = x_i + alpha * self.attractiveness(x_i, x_j) * (x_j - x_i) + r
        return np.clip(x_new, -5.0, 5.0)

    def __call__(self, func):
        fireflies = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))

        for i in range(self.budget):
            for j in range(self.budget):
                if func(fireflies[j]) < func(fireflies[i]):
                    alpha = 0.9 - i / self.budget  # Adaptive alpha value
                    fireflies[i] = self.move_firefly(fireflies[i], fireflies[j], alpha)

            f = func(fireflies[i])
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = fireflies[i]

        return self.f_opt, self.x_opt