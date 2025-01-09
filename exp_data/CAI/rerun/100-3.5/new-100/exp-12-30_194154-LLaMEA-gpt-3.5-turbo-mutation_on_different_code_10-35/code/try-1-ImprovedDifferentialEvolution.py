class ImprovedDifferentialEvolution(DifferentialEvolution):
    def __init__(self, budget=10000, dim=10, F_min=0.2, F_max=0.8, CR_min=0.5, CR_max=1.0, pop_size=20):
        super().__init__(budget, dim)
        self.F_min = F_min
        self.F_max = F_max
        self.CR_min = CR_min
        self.CR_max = CR_max
        self.F = F_min
        self.CR = CR_min
        self.pop_size = pop_size

    def __call__(self, func):
        bounds = func.bounds
        population = np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))

        for i in range(self.budget):
            for j in range(self.pop_size):
                idxs = list(range(self.pop_size))
                idxs.remove(j)
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[j])

                f_trial = func(trial)
                if f_trial < func(population[j]):
                    population[j] = trial

                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

            # Adaptive control of F and CR based on performance
            successful_mutations = np.sum(f_trial < func(population[j]) for j, f_trial in enumerate(f_trial))
            mutation_rate = successful_mutations / self.pop_size

            if mutation_rate > 0.2:
                self.F = min(self.F_max, self.F * 1.2)
            elif mutation_rate < 0.2:
                self.F = max(self.F_min, self.F / 1.2)

            crossover_rate = successful_mutations / self.pop_size
            if crossover_rate > 0.5:
                self.CR = min(self.CR_max, self.CR * 1.1)
            elif crossover_rate < 0.5:
                self.CR = max(self.CR_min, self.CR / 1.1)

        return self.f_opt, self.x_opt