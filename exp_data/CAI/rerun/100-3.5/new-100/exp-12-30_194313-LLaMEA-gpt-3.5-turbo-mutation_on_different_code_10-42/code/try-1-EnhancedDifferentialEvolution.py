class EnhancedDifferentialEvolution(DifferentialEvolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.F_min = 0.2
        self.F_max = 0.8
        self.CR_min = 0.2
        self.CR_max = 0.8

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])
        F_array = np.random.uniform(self.F_min, self.F_max, size=self.budget)
        CR_array = np.random.uniform(self.CR_min, self.CR_max, size=self.budget)
        
        for i in range(self.budget):
            for j in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != j]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = population[a] + F_array[j] * (population[b] - population[c])
                crossover_mask = np.random.rand(self.dim) < CR_array[j]
                trial = np.where(crossover_mask, mutant, population[j])
                
                f_trial = func(trial)
                if f_trial < fitness[j]:
                    population[j] = trial
                    fitness[j] = f_trial
                
                if fitness[j] < self.f_opt:
                    self.f_opt = fitness[j]
                    self.x_opt = population[j]
        
        return self.f_opt, self.x_opt