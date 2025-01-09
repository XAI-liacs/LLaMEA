class ImprovedDifferentialEvolution(DifferentialEvolution):
    def __init__(self, adapt_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.adapt_rate = adapt_rate
        self.adaptive_F = self.F

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))

        for _ in range(self.budget):
            new_population = np.zeros_like(population)
            for i in range(self.pop_size):
                indices = np.random.choice(range(self.pop_size), 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + self.adaptive_F * (b - c), bounds[0], bounds[1])
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[i])
                f_trial = func(trial)
                if f_trial < func(population[i]):
                    population[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

            if np.random.rand() < self.adapt_rate:
                self.adaptive_F = np.clip(self.F + np.random.normal(0, 0.1), 0, 1)

        return self.f_opt, self.x_opt