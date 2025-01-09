class DE:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=30, F_l=0.1, F_u=0.9, F_decay=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.F_l = F_l
        self.F_u = F_u
        self.F_decay = F_decay
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        pop_fitness = np.array([func(x) for x in pop])
        
        for i in range(self.budget):
            for j in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                F_val = np.random.uniform(self.F_l, self.F_u) * self.F
                mutant = np.clip(a + F_val * (b - c), func.bounds.lb, func.bounds.ub)
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, pop[j])
                f = func(trial)
                
                if f < pop_fitness[j]:
                    pop[j] = trial
                    pop_fitness[j] = f
                
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
                    
                self.F *= self.F_decay
          
        return self.f_opt, self.x_opt