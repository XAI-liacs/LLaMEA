class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, population_size=50, F_min=0.2, F_max=0.8, F_decay=0.99):
        self.budget = budget
        self.dim = dim
        self.F = F  # Differential weight
        self.CR = CR  # Crossover rate
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.F_min = F_min
        self.F_max = F_max
        self.F_decay = F_decay

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.population_size, self.dim))
        
        for _ in range(self.budget):
            self.F = max(self.F * self.F_decay, self.F_min)
            for i in range(self.population_size):
                idxs = list(range(self.population_size))
                idxs.remove(i)
                a, b, c = np.random.choice(idxs, 3, replace=False)
                
                mutant = population[a] + self.F * (population[b] - population[c])
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[i])
                
                f = func(trial)
                if f < func(population[i]):
                    population[i] = trial
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial
        
        return self.f_opt, self.x_opt