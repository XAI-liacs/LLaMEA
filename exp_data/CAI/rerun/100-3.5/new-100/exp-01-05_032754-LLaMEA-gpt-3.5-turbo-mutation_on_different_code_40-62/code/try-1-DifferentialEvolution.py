class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_min=0.2, F_max=0.8, CR_min=0.1, CR_max=0.9):
        self.budget = budget
        self.dim = dim
        self.F = np.full(budget, F)
        self.CR = np.full(budget, CR)
        self.F_min = F_min
        self.F_max = F_max
        self.CR_min = CR_min
        self.CR_max = CR_max
        self.f_opt = np.Inf
        self.x_opt = None
    
    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        
        for i in range(self.budget):
            for j in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != i and idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                F_i = np.clip(np.random.normal(self.F[i], 0.1), self.F_min, self.F_max)
                CR_i = np.clip(np.random.normal(self.CR[i], 0.1), self.CR_min, self.CR_max)
                
                mutant = np.clip(a + F_i * (b - c), func.bounds.lb, func.bounds.ub)
                trial = np.where(np.random.rand(self.dim) < CR_i, mutant, population[i])
                
                f = func(trial)
                if f < func(population[i]):
                    population[i] = trial
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial
                        
        return self.f_opt, self.x_opt