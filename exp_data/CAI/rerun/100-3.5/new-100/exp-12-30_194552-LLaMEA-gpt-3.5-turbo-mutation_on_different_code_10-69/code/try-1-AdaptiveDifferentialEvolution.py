class AdaptiveDifferentialEvolution(DifferentialEvolution):
    def __init__(self, budget=10000, dim=10, population_size=50):
        super().__init__(budget, dim, population_size=population_size)
        self.F_min = 0.2
        self.F_max = 0.8
        self.CR_min = 0.3
        self.CR_max = 0.9

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.population_size, self.dim))
        F_list = np.random.uniform(self.F_min, self.F_max, size=self.population_size)
        CR_list = np.random.uniform(self.CR_min, self.CR_max, size=self.population_size)

        for i in range(self.budget):
            for j in range(self.population_size):
                target = population[j]
                indices = [idx for idx in range(self.population_size) if idx != j]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                F = F_list[j]
                CR = CR_list[j]

                mutant = np.clip(a + F * (b - c), func.bounds.lb, func.bounds.ub)

                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, target)

                f = func(trial)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
                    population[j] = trial

                # Adaptive control over F and CR
                if np.random.rand() < 0.1:
                    F_list[j] = np.clip(F + np.random.normal(0, 0.1), self.F_min, self.F_max)
                    CR_list[j] = np.clip(CR + np.random.normal(0, 0.1), self.CR_min, self.CR_max)

        return self.f_opt, self.x_opt