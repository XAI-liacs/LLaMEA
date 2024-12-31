class ImprovedFireflyAlgorithm(FireflyAlgorithm):
    def __init__(self, budget=10000, dim=10, alpha_init=0.2, beta0=1.0, gamma=0.1, alpha_decay=0.9):
        super().__init__(budget, dim, alpha_init, beta0, gamma)
        self.alpha_decay = alpha_decay

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))

        for i in range(self.budget):
            for j in range(self.budget):
                if func(fireflies[j]) < func(fireflies[i]):  # Minimization problem
                    beta = self.attractiveness(fireflies[i], fireflies[j])
                    fireflies[i] = self.move_firefly(fireflies[i], fireflies[j], beta)

            f = func(fireflies[i])
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = fireflies[i]
            
            self.alpha *= self.alpha_decay  # Dynamic alpha adjustment
            
        return self.f_opt, self.x_opt

    def move_firefly(self, xi, xj, beta):
        r = np.linalg.norm(xi - xj)
        return xi + self.alpha * (beta * (xj - xi)) + 0.01 * np.random.randn(self.dim)