class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha_min=0.1, alpha_max=0.9, beta0=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.beta0 = beta0
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        def attractiveness(r, alpha):
            return alpha * np.exp(-self.beta0 * r**2)

        lb = func.bounds.lb
        ub = func.bounds.ub

        fireflies = np.random.uniform(lb, ub, size=(self.budget, self.dim))
        f_vals = np.apply_along_axis(func, 1, fireflies)

        for i in range(self.budget):
            for j in range(self.budget):
                if f_vals[i] > f_vals[j]:
                    alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * (i / self.budget)
                    beta = self.beta0 * np.exp(-alpha * np.linalg.norm(fireflies[i] - fireflies[j])**2)
                    fireflies[i] += beta * (fireflies[j] - fireflies[i]) + 0.01 * np.random.randn(self.dim)
                    fireflies[i] = np.clip(fireflies[i], lb, ub)
                    f_vals[i] = func(fireflies[i])

        best_idx = np.argmin(f_vals)
        self.f_opt = f_vals[best_idx]
        self.x_opt = fireflies[best_idx]

        return self.f_opt, self.x_opt