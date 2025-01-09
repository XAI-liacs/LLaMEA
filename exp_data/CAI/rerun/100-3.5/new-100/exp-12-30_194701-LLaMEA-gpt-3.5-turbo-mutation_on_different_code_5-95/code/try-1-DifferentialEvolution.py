class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=30, adapt_rate=0.05):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.adapt_rate = adapt_rate
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        def mutation(population, target_idx):
            candidates = population[[idx for idx in range(len(population)) if idx != target_idx]]
            a, b, c = np.random.choice(len(candidates), 3, replace=False)
            f_adapt = np.clip(np.random.normal(self.F, self.adapt_rate), 0, 2)
            mutant = population[a] + f_adapt * (population[b] - population[c])
            return mutant
        
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        
        for i in range(self.budget):
            for j in range(self.pop_size):
                x = population[j]
                mutant = mutation(population, j)
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, x)
                f = func(trial)
                
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
                population[j] = trial
        
        return self.f_opt, self.x_opt