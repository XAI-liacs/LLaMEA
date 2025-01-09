class AdaptiveFireflyAlgorithm(FireflyAlgorithm):
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta0=1.0, gamma=1.0):
        super().__init__(budget, dim, alpha, beta0, gamma)
        
    def move_fireflies(self, x, f, func):
        for i in range(self.budget):
            for j in range(self.budget):
                if f[j] < f[i]:
                    beta = self.beta0 * np.exp(-self.alpha * np.square(np.linalg.norm(x[j] - x[i])))
                    gamma_i = self.gamma / (1 + self.alpha * np.square(np.linalg.norm(x[j] - x[i])))  # Adaptive gamma
                    x[i] += beta * (x[j] - x[i]) + np.random.uniform(-1, 1, self.dim)
                    x[i] = np.clip(x[i], func.bounds.lb, func.bounds.ub)
                    f[i] = func(x[i])
                    if f[i] < self.f_opt:
                        self.f_opt = f[i]
                        self.x_opt = x[i]
        return self.f_opt, self.x_opt