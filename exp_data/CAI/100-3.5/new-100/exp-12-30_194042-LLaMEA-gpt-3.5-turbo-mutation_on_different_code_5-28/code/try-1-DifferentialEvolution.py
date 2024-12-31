class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_min=0.2, F_max=0.8, CR_min=0.1, CR_max=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.F_min = F_min
        self.F_max = F_max
        self.CR_min = CR_min
        self.CR_max = CR_max
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for i in range(self.budget):
            for j in range(len(population)):
                idxs = [idx for idx in range(len(population)) if idx != j]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                F_val = np.clip(self.F + 0.1 * (self.f_opt - fitness[j]), self.F_min, self.F_max)
                CR_val = np.clip(self.CR + 0.1 * (self.f_opt - fitness[j]), self.CR_min, self.CR_max)
                mutant = np.clip(population[a] + F_val * (population[b] - population[c]), func.bounds.lb, func.bounds.ub)
                crossover = np.random.rand(self.dim) < CR_val
                trial = np.where(crossover, mutant, population[j])
                
                f_trial = func(trial)
                if f_trial < fitness[j]:
                    fitness[j] = f_trial
                    population[j] = trial
                    
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

        return self.f_opt, self.x_opt