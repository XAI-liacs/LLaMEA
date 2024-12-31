class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F_init=0.5, F_decay=0.95, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F_init
        self.F_decay = F_decay
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.dim, self.dim))
        
        for i in range(self.budget):
            for j in range(self.dim):
                indices = [idx for idx in range(self.dim) if idx != j]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
                
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[j])
                
                f_trial = func(trial)
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
                    population[j] = trial
            
            self.F *= self.F_decay
                
        return self.f_opt, self.x_opt