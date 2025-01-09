class ImprovedDifferentialEvolution(DifferentialEvolution):
    def __init__(self, budget=10000, dim=10):
        super().__init__(budget, dim)
        self.CR = 0.9  # Crossover rate
        self.F = 0.5   # Differential weight
        self.strategy = 'rand-to-best/1/bin'  # Enhanced mutation strategy

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for i in range(self.budget):
            for j in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                if self.strategy == 'rand-to-best/1/bin':
                    best_idx = np.argmin(fitness)
                    best = population[best_idx]
                    mutant = np.clip(population[j] + self.F * (best - population[j]) + self.F * (a - b), func.bounds.lb, func.bounds.ub)
                
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[j])
                
                f_trial = func(trial)
                if f_trial < fitness[j]:
                    population[j] = trial
                    fitness[j] = f_trial
                    
                if fitness[j] < self.f_opt:
                    self.f_opt = fitness[j]
                    self.x_opt = population[j]

        return self.f_opt, self.x_opt