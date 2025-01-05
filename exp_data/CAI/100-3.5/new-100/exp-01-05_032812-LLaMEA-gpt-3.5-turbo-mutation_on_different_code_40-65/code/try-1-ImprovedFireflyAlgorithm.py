class ImprovedFireflyAlgorithm(FireflyAlgorithm):
    def __init__(self, budget=10000, dim=10, alpha_init=0.5, gamma_init=0.1):
        super().__init__(budget, dim)
        self.alpha_init = alpha_init
        self.gamma_init = gamma_init

    def dynamic_params(self, t):
        alpha_t = self.alpha_init * np.exp(-0.1 * t)
        gamma_t = self.gamma_init * np.exp(-0.1 * t)
        return alpha_t, gamma_t

    def __call__(self, func):
        x = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        f = np.array([func(x_i) for x_i in x])

        for t in range(self.budget):
            alpha_t, gamma_t = self.dynamic_params(t)
            for i in range(self.budget):
                for j in range(self.budget):
                    if f[j] < f[i]:
                        r = np.linalg.norm(x[i] - x[j])
                        beta = self.attractiveness(r)
                        x[i] += alpha_t * (x[j] - x[i]) * beta
                        f[i] = func(x[i])
                        if f[i] < self.f_opt:
                            self.f_opt = f[i]
                            self.x_opt = x[i]

        return self.f_opt, self.x_opt