class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F_init=0.8, CR=0.9, pop_size=50, F_min=0.2, F_max=0.8):
        self.budget = budget
        self.dim = dim
        self.F_init = F_init
        self.F = F_init
        self.CR = CR
        self.pop_size = pop_size
        self.F_min = F_min
        self.F_max = F_max
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        
        for i in range(self.budget):
            for j in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != j]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                F = self.F_min + (self.F_max - self.F_min) * (i / self.budget)  # Dynamic adaptation of F
                mutant = pop[a] + F * (pop[b] - pop[c])
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, pop[j])
                
                f_trial = func(trial)
                
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
                    pop[j] = trial
        
        return self.f_opt, self.x_opt