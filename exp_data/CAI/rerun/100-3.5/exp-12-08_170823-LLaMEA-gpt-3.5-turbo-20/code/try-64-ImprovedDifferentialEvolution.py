class ImprovedDifferentialEvolution(DifferentialEvolution):
    def __init__(self, budget=10000, dim=10):
        super().__init__(budget, dim)
        self.diverse_ratio = 0.1  # Ratio of diverse individuals
        self.diverse_iters = int(self.budget * self.diverse_ratio)

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for i in range(self.budget):
            for j in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != j]
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
            
            # Ensure diversity by randomly replacing worst individuals
            if i % self.diverse_iters == 0:
                sorted_idxs = np.argsort(fitness)
                for k in range(int(self.budget * self.diverse_ratio)):
                    random_idx = np.random.randint(int(self.budget * 0.8), self.budget)
                    population[random_idx] = np.random.uniform(func.bounds.lb, func.bounds.ub)
                    fitness[random_idx] = func(population[random_idx])

        return self.f_opt, self.x_opt