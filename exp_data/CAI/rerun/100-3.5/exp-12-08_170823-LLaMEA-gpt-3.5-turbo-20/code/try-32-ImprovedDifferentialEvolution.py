class ImprovedDifferentialEvolution(DifferentialEvolution):
    def __init__(self, budget=10000, dim=10):
        super().__init__(budget, dim)
        self.NP = int(5 * dim) # Dynamic population size based on dimensionality
        
    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.NP, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for i in range(self.budget):
            for j in range(self.NP):
                idxs = [idx for idx in range(self.NP) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
                
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