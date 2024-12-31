class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_range=(0.1, 0.9), CR_range=(0.1, 0.9), adapt_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.F_range = F_range
        self.CR_range = CR_range
        self.adapt_rate = adapt_rate
        self.f_opt = np.Inf
        self.x_opt = None

    def adapt_parameters(self, diversity):
        self.F = max(self.F_range[0], min(self.F_range[1], self.F + np.random.normal(0, self.adapt_rate)))
        self.CR = max(self.CR_range[0], min(self.CR_range[1], self.CR + np.random.normal(0, self.adapt_rate)))

    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.budget, self.dim))

        for _ in range(self.budget):
            new_pop = np.empty_like(pop)
            diversity = np.std(pop, axis=0)

            for i in range(len(pop)):
                self.adapt_parameters(diversity)
                trial_vector = self.mutate(pop, i)
                trial = self.crossover(pop[i], trial_vector)
                new_pop[i], f_trial = self.select(func, pop[i], trial)

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = new_pop[i]

            pop = new_pop

        return self.f_opt, self.x_opt