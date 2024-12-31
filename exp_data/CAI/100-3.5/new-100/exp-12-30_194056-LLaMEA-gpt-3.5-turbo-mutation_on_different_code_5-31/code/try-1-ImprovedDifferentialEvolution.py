class ImprovedDifferentialEvolution(DifferentialEvolution):
    def __init__(self, budget=10000, dim=10):
        super().__init__(budget, dim)
        self.F_min = 0.2
        self.F_max = 0.8
        self.CR_min = 0.1
        self.CR_max = 0.9
        self.F = np.full(self.budget, self.F_max)
        self.CR = np.full(self.budget, self.CR_max)

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            target_idx = np.random.randint(self.budget)
            indices = np.random.choice(np.delete(np.arange(self.budget), target_idx, axis=0), size=2, replace=False)
            a, b, c = population[indices]
            self.F[i] = max(self.F_min, min(self.F_max, np.random.normal(self.F[i], 0.1)))
            self.CR[i] = max(self.CR_min, min(self.CR_max, np.random.normal(self.CR[i], 0.1)))
            mutant = population[target_idx] + self.F[i] * (a - b)
            crossover = np.random.rand(self.dim) < self.CR[i]
            trial = np.where(crossover, mutant, population[target_idx])

            f_target = func(population[target_idx])
            f_trial = func(trial)
            if f_trial < f_target:
                population[target_idx] = trial
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

        return self.f_opt, self.x_opt