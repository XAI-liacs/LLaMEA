class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=20, F_min=0.2, F_max=0.8):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.F_min = F_min
        self.F_max = F_max
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        for i in range(self.budget):
            F = np.clip(self.F + 0.01 * np.random.randn(), self.F_min, self.F_max)
            for j in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = pop[a] + F * (pop[b] - pop[c])
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, pop[j])
                
                f = func(pop[j])
                f_trial = func(trial)
                if f_trial < f:
                    pop[j] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        
        return self.f_opt, self.x_opt