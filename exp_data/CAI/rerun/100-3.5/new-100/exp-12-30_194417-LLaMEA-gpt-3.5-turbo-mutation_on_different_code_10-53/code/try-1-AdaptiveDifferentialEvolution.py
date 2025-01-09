class AdaptiveDifferentialEvolution(DifferentialEvolution):
    def __init__(self, budget=10000, dim=10, F_init=0.5, CR_init=0.9, F_decay=0.95, CR_growth=0.05):
        super().__init__(budget, dim, F_init, CR_init)
        self.F_decay = F_decay
        self.CR_growth = CR_growth

    def __call__(self, func):
        pop_size = 10 * self.dim
        bounds = np.array([func.bounds.lb, func.bounds.ub])

        population = np.random.uniform(bounds[0], bounds[1], size=(pop_size, self.dim))
        F = self.F
        CR = self.CR

        for i in range(self.budget):
            for j in range(pop_size):
                idxs = np.arange(pop_size)
                idxs = idxs[idxs != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                mutant = np.clip(a + F * (b - c), bounds[0], bounds[1])

                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, population[j])

                f_trial = func(trial)
                if f_trial < func(population[j]):
                    population[j] = trial

                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

            F *= self.F_decay
            CR = min(1.0, CR + self.CR_growth)

        return self.f_opt, self.x_opt