class ImprovedDifferentialEvolution(DifferentialEvolution):
    def __init__(self, budget=10000, dim=10, F_init=0.5, CR_init=0.9, pop_size=50, F_decay=0.9, CR_raise=0.1):
        super().__init__(budget, dim, F_init, CR_init, pop_size)
        self.F_init = F_init
        self.CR_init = CR_init
        self.F_decay = F_decay
        self.CR_raise = CR_raise

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        F = self.F_init
        CR = self.CR_init
        for i in range(self.budget):
            self.F = max(0.1, F * self.F_decay)
            self.CR = min(1.0, CR + self.CR_raise)
            population = self.evolve_population(population, func)
            best_idx = np.argmin([func(ind) for ind in population])
            if func(population[best_idx]) < self.f_opt:
                self.f_opt = func(population[best_idx])
                self.x_opt = population[best_idx]

        return self.f_opt, self.x_opt