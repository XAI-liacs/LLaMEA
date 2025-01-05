class DifferentialEvolutionImproved:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F  
        self.CR = CR  
        self.f_opt = np.Inf
        self.x_opt = None
        self.F_l = 0.1
        self.F_u = 0.9
        self.CR_l = 0.1
        self.CR_u = 0.9

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            indices = np.arange(self.budget)
            indices = indices[indices != i]
            a, b, c = np.random.choice(indices, 3, replace=False)

            F_val = np.random.uniform(self.F_l, self.F_u)
            CR_val = np.random.uniform(self.CR_l, self.CR_u)

            mutant = population[a] + F_val * (population[b] - population[c])
            crossover_mask = np.random.rand(self.dim) < CR_val
            offspring = np.where(crossover_mask, mutant, population[i])

            f_offspring = func(offspring)
            if f_offspring < func(population[i]):
                population[i] = offspring
            
            if f_offspring < self.f_opt:
                self.f_opt = f_offspring
                self.x_opt = offspring

        return self.f_opt, self.x_opt