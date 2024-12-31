class ImprovedHEDHS(HEDHS):
    def adaptive_parameter_tuning(self, func, population):
        for i in range(self.budget):
            self.hmcr = max(0.7 - 0.5 * i / self.budget, 0.1)
            self.par = min(0.5 + 0.5 * i / self.budget, 0.9)
            self.bw = max(0.01 - 0.01 * i / self.budget, 0.001)
            self.harmony_search(func, population)
        return self.f_opt, self.x_opt

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))

        # Perform hybrid optimization with adaptive parameter tuning
        self.adaptive_parameter_tuning(func, population)

        return self.f_opt, self.x_opt