class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.9, F_min=0.2, F_max=0.8, F_decay=0.95):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.F_min = F_min
        self.F_max = F_max
        self.F_decay = F_decay
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        f_vals = np.array([func(x) for x in population])
        
        for _ in range(self.budget):
            self.F = max(self.F * self.F_decay, self.F_min)
            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                mutant = population[a] + self.F * (population[b] - population[c])
                crossover = np.random.rand(self.dim) < self.CR
                
                trial = np.where(crossover, mutant, population[i])
                f_trial = func(trial)
                
                if f_trial < f_vals[i]:
                    population[i] = trial
                    f_vals[i] = f_trial
                
                if f_vals[i] < self.f_opt:
                    self.f_opt = f_vals[i]
                    self.x_opt = population[i]
                    
        return self.f_opt, self.x_opt