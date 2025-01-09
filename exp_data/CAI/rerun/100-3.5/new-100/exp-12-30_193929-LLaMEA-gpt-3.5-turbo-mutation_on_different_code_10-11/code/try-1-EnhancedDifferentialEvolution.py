class EnhancedDifferentialEvolution(DifferentialEvolution):
    def __init__(self, budget=10000, dim=10, f=0.5, cr=0.9, f_decay=0.9, cr_decay=0.9):
        super().__init__(budget, dim, f, cr)
        self.f_decay = f_decay
        self.cr_decay = cr_decay

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            for j in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != i and idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                # Dynamic adaptation of mutation factor and crossover rate
                f_dynamic = self.f * np.abs(np.random.normal(0, 1, self.dim))
                cr_dynamic = self.cr * np.abs(np.random.normal(0, 1, self.dim))

                mutant = population[i] + f_dynamic * (a - b)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                crossover = np.random.rand(self.dim) < cr_dynamic
                trial = np.where(crossover, mutant, population[i])

                f_trial = func(trial)
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
                    population[i] = trial

            # Decay mutation factor and crossover rate
            self.f *= self.f_decay
            self.cr *= self.cr_decay

        return self.f_opt, self.x_opt