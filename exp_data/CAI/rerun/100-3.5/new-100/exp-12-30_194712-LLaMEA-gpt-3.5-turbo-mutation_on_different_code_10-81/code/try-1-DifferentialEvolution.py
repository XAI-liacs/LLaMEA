class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_min=0.2, F_max=0.8, F_decay=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.F_min = F_min
        self.F_max = F_max
        self.F_decay = F_decay
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.dim, self.dim))
        
        for _ in range(self.budget):
            self.F = max(self.F * self.F_decay, self.F_min)
            for i in range(self.dim):
                a, b, c = np.random.choice(population, 3, replace=False)
                mutant = a + self.F * (b - c)
                crossover_prob = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_prob, mutant, population[i])
                
                f_trial = func(trial)
                if f_trial < func(population[i]):
                    population[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
        
        return self.f_opt, self.x_opt