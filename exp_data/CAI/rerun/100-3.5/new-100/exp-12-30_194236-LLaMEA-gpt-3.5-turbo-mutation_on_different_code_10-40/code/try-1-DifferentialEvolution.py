class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_min=0.2, F_max=0.8, scaling_factor=0.1):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.F_min = F_min
        self.F_max = F_max
        self.scaling_factor = scaling_factor
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop_size = 10 * self.dim
        pop = np.random.uniform(-5, 5, (pop_size, self.dim))
        
        for _ in range(self.budget):
            for i in range(pop_size):
                target = pop[i]
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                F = np.clip(np.random.normal(self.F, self.scaling_factor), self.F_min, self.F_max)
                mutant = np.clip(a + F * (b - c), -5, 5)
                jrand = np.random.randint(self.dim)
                trial = [mutant[j] if (np.random.rand() < self.CR or j == jrand) else target[j] for j in range(self.dim)]
                
                f = func(trial)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
        
        return self.f_opt, self.x_opt