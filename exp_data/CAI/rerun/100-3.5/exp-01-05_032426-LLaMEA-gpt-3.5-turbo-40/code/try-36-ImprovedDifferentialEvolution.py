class ImprovedDifferentialEvolution(DifferentialEvolution):
    def __init__(self, budget=10000, dim=10, F_init=0.5, CR_init=0.9, F_decay=0.9, CR_growth=0.1):
        super().__init__(budget, dim, F_init, CR_init)
        self.F_decay = F_decay
        self.CR_growth = CR_growth

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        F = self.F
        CR = self.CR

        for i in range(self.budget):
            indices = np.arange(self.budget)
            indices = indices[indices != i]
            a, b, c = np.random.choice(indices, 3, replace=False)

            mutant = population[a] + F * (population[b] - population[c])
            crossover_mask = np.random.rand(self.dim) < CR
            offspring = np.where(crossover_mask, mutant, population[i])

            f_offspring = func(offspring)
            if f_offspring < func(population[i]):
                population[i] = offspring
                F *= self.F_decay
                CR += self.CR_growth

            if f_offspring < self.f_opt:
                self.f_opt = f_offspring
                self.x_opt = offspring

        return self.f_opt, self.x_opt