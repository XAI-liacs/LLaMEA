class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_l=0.1, F_u=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.F_l = F_l  # Lower bound of F
        self.F_u = F_u  # Upper bound of F
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.dim, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for _ in range(self.budget):
            self.F = np.clip(np.random.normal(self.F, 0.1), self.F_l, self.F_u)
            for i in range(self.dim):
                a, b, c = np.random.choice(np.delete(np.arange(self.dim), i), 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
                idx = np.random.rand(self.dim) < self.CR
                trial = np.where(idx, mutant, population[i])
                
                f = func(trial)
                if f < fitness[i]:
                    fitness[i] = f
                    population[i] = trial
                    
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial
            
        return self.f_opt, self.x_opt