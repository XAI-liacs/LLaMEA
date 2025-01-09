class ImprovedFireflyAlgorithm(FireflyAlgorithm):
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta0=1.0):
        super().__init__(budget, dim, alpha, beta0)
    
    def dynamic_params(self, iter_num):
        self.alpha = 0.2 + 0.5 * np.exp(-0.01 * iter_num)
        self.beta0 = 1.0 / (1.0 + 0.01 * iter_num)
    
    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))

        for i in range(self.budget):
            self.dynamic_params(i)
            
            for j in range(self.budget):
                if func(fireflies[j]) < func(fireflies[i]):
                    fireflies[i] = self.move_firefly(fireflies[i], fireflies[j], self.attractiveness(np.linalg.norm(fireflies[i] - fireflies[j])))

            f = func(fireflies[i])
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = fireflies[i]

        return self.f_opt, self.x_opt