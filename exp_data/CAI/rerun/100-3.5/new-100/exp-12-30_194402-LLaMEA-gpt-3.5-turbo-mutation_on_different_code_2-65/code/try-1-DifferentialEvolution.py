class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=20, F_decay=0.95):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.F_decay = F_decay
        self.f_opt = np.Inf
        self.x_opt = None
        
    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        
        for _ in range(self.budget):
            self.F *= self.F_decay
            for i in range(self.pop_size):
                idxs = np.random.choice(np.setdiff1d(np.arange(self.pop_size), i, assume_unique=True), 3, replace=False)
                a, b, c = pop[idxs]
                
                mutant = pop[a] + self.F * (pop[b] - pop[c])
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, pop[i])
                
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial
            
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.f_opt:
                self.f_opt = fitness[best_idx]
                self.x_opt = pop[best_idx]
        
        return self.f_opt, self.x_opt