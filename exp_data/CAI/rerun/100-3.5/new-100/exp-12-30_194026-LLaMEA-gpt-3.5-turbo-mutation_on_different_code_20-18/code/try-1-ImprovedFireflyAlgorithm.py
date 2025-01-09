class ImprovedFireflyAlgorithm(FireflyAlgorithm):
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta0=1.0, gamma=1.0, step_size=0.1, population_size=20):
        super().__init__(budget, dim, alpha, beta0, gamma)
        self.step_size = step_size
        self.population_size = population_size

    def move_firefly(self, x, best_x, func):
        r = np.random.uniform(-1, 1, size=self.dim)
        beta = self.beta0 * np.exp(-self.gamma * self.alpha)
        new_x = x + self.attractiveness(x, best_x, func) * (best_x - x) + beta * r
        new_x += self.step_size * np.random.uniform(-1, 1, size=self.dim)  # Adaptive step size
        return np.clip(new_x, func.bounds.lb, func.bounds.ub)

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.population_size, self.dim))

        for _ in range(self.budget):
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if func(fireflies[j]) < func(fireflies[i]):
                        fireflies[i] = self.move_firefly(fireflies[i], fireflies[j], func)

            f_values = [func(firefly) for firefly in fireflies]
            best_index = np.argmin(f_values)
            if f_values[best_index] < self.f_opt:
                self.f_opt = f_values[best_index]
                self.x_opt = fireflies[best_index]

        return self.f_opt, self.x_opt