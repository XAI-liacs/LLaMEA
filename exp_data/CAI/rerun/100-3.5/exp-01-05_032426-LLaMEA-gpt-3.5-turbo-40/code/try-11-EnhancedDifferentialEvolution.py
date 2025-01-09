class EnhancedDifferentialEvolution(DifferentialEvolution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.F_lower = 0.2
        self.F_upper = 0.8

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))

        for i in range(self.budget):
            indices = np.arange(self.budget)
            indices = indices[indices != i]
            a, b, c = np.random.choice(indices, 3, replace=False)

            F_current = np.random.uniform(self.F_lower, self.F_upper)
            mutant = population[a] + F_current * (population[b] - population[c])
            crossover_mask = np.random.rand(self.dim) < self.CR
            offspring = np.where(crossover_mask, mutant, population[i])

            f_offspring = func(offspring)
            if f_offspring < func(population[i]):
                population[i] = offspring

            if f_offspring < self.f_opt:
                self.f_opt = f_offspring
                self.x_opt = offspring

        return self.f_opt, self.x_opt