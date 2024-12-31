class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, NP=30, F_min=0.2, F_max=0.8, F_decay=0.95):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.NP = NP
        self.F_min = F_min
        self.F_max = F_max
        self.F_decay = F_decay
        self.f_opt = np.Inf
        self.x_opt = None
        
    def evolve_population(self, population, func):
        for i in range(self.NP):
            target = population[i]
            indices = [idx for idx in range(self.NP) if idx != i]
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            F = np.clip(self.F * np.exp(-self.F_decay * i), self.F_min, self.F_max)
            mutant = np.clip(a + F * (b - c), func.bounds.lb, func.bounds.ub)
            
            crossover_points = np.random.rand(self.dim) < self.CR
            trial = np.where(crossover_points, mutant, target)
            
            f_target = func(target)
            f_trial = func(trial)
            if f_trial < f_target:
                population[i] = trial
        
        return population