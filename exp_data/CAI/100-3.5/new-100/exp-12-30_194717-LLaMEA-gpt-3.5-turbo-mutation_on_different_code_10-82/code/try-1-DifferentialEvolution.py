class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F_init=0.5, F_decay=0.9, CR=0.9, population_size=30):
        self.budget = budget
        self.dim = dim
        self.F_init = F_init
        self.F_decay = F_decay
        self.CR = CR
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.population_size, self.dim))
        F = self.F_init

        for _ in range(self.budget):
            for i in range(self.population_size):
                target = population[i]
                candidates = [ind for ind in population if ind is not target]
                idxs = np.random.choice(len(candidates), 3, replace=False)
                a, b, c = candidates[idxs]
                mutant = a + F * (b - c)
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, target)
                
                f = func(trial)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
            
            F *= self.F_decay
        
        return self.f_opt, self.x_opt