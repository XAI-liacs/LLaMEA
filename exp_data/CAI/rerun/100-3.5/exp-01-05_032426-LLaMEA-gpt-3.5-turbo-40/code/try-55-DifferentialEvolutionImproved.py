class DifferentialEvolutionImproved:
    def __init__(self, budget=10000, dim=10, F_init=0.5, CR_init=0.9):
        self.budget = budget
        self.dim = dim
        self.F_init = F_init  # Initial F value
        self.CR_init = CR_init  # Initial CR value
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        F = self.F_init
        CR = self.CR_init
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

            if f_offspring < self.f_opt:
                self.f_opt = f_offspring
                self.x_opt = offspring

            # Adaptation of F and CR values
            F = max(0.1, F * 0.95) if np.random.rand() > 0.1 else np.random.uniform(0.1, 0.9)
            CR = max(0.1, CR * 0.95) if np.random.rand() > 0.1 else np.random.uniform(0.1, 0.9)

        return self.f_opt, self.x_opt