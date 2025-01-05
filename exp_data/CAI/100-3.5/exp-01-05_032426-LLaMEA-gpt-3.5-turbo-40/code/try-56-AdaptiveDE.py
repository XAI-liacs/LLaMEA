class AdaptiveDE(DifferentialEvolution):
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_adapt=0.1, CR_adapt=0.1):
        super().__init__(budget, dim, F, CR)
        self.F_adapt = F_adapt
        self.CR_adapt = CR_adapt

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            indices = np.arange(self.budget)
            indices = indices[indices != i]
            a, b, c = np.random.choice(indices, 3, replace=False)

            F_i = np.clip(np.random.normal(self.F, self.F_adapt), 0, 2)  # Adaptive F
            CR_i = np.clip(np.random.normal(self.CR, self.CR_adapt), 0, 1)  # Adaptive CR

            mutant = population[a] + F_i * (population[b] - population[c])
            crossover_mask = np.random.rand(self.dim) < CR_i
            offspring = np.where(crossover_mask, mutant, population[i])

            f_offspring = func(offspring)
            if f_offspring < func(population[i]):
                population[i] = offspring
            
            if f_offspring < self.f_opt:
                self.f_opt = f_offspring
                self.x_opt = offspring

        return self.f_opt, self.x_opt